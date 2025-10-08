import pandas as pd
import numpy as np
import warnings
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict
from sentence_transformers import SentenceTransformer, util


def read_any_file_type(file: Path) -> pd.DataFrame:
    """
    Reads a file from a filepath
    """

    if file.name.endswith("xls") or file.name.endswith("xlsx"):
        df = pd.read_excel(file, sheet_name=None)
    elif file.name.endswith("parquet"):
        df = pd.read_parquet(file)
    elif file.name.endswith("csv"):
        df = pd.read_csv(file)
    elif file.name.endswith("txt") or file.name.endswith("tsv"):
        df = pd.read_csv(file, sep="\t")
    else:
        msg = f"File Extension {file.name.split('.')[1]} not found."
        warnings.warn(msg)

    return df


def read_files(
    _path_: str,
    name_dicts: dict = {
        "payments": "agg_payments",
        "orders": "agg_orders",
        "returns": "agg_returns",
        "mfn_returns": "agg_returns-mfn",
        "replacements": "agg_replacements",
        "shipments": "agg_shipments",
        "ratecard": "rate_card",
        "category_mapper": "commission_category_mapper",
        "weight_mapper": "weight_mapper",
        "postcode_mapper": "postcode",
        "mtr": "mtr",
        "size_band_mapper": "size_band_mapper",
        "category_mapper_fixed_closing": "fixed_closing_category_mapper",
        "fulfilment model mapper": "fultilment_model_mapper",
    },
) -> dict:
    """
    Read relevant files for conducting fee audit
    """

    data_dict = defaultdict(pd.DataFrame)
    folder_path = Path(_path_)

    for file in folder_path.iterdir():
        if file.name == ".DS_Store":
            continue
        else:
            matched = [n for n in name_dicts if name_dicts.get(n) in file.name]
            best_match = max(matched, key=len) if matched else None
            print(file.name)
            print(f"Matches found: {matched}")
            print(f"Best match: {best_match}")
            df = read_any_file_type(file)

            if best_match in data_dict:
                temp_df = data_dict[best_match]
                df = pd.concat([temp_df, df], axis=0)

            data_dict[best_match] = df

    data_dict = dict(data_dict)

    return data_dict


def rename_cols(data_dict: dict) -> dict:
    """
    Renames and standardizes names for critical columns across all the data sources
    """

    rename_map = {
        "amazon-order-id": "order-id",
        "original-amazon-order-id": "order-id",
    }

    for file, df in data_dict.items():
        if file not in ["ratecard", "postcode_mapper"]:
            df.rename(columns=rename_map, inplace=True)
            data_dict[file] = df

    return data_dict


def create_keys(
    df: pd.DataFrame, key_cols: list = ["order-id", "asin"]
) -> pd.DataFrame:
    """
    Creates a mapping key
    """
    df["key"] = df[key_cols].astype(str).agg("@".join, axis=1)

    return df


def add_asin_to_pmt(data_dict: dict) -> dict:
    """
    Adds ASIN information in the payment file.
    """

    order = data_dict["orders"].copy()
    pmt = data_dict["payments"].copy()

    asin_mapper_dict = order.groupby("sku")["asin"].first().to_dict()
    pmt["asin"] = pmt["sku"].map(asin_mapper_dict)
    pmt["asin"] = pmt["asin"].fillna("Not-found")
    data_dict["asin_mapper"] = asin_mapper_dict
    data_dict["payments"] = pmt

    return data_dict


def forward_fill_settlement_date(
    data_dict,
    cols_to_ffill: list = [
        "settlement-start-date",
        "settlement-end-date",
        "deposit-date",
    ],
) -> dict:
    """
    Forward fills data in missing columns
    """
    pmt = data_dict["payments"]
    pmt[cols_to_ffill] = pmt[cols_to_ffill].ffill()
    data_dict["payments"] = pmt

    return data_dict


def create_fee_table(data_dict: dict, req_fee_types: list) -> dict:
    """
    Filters the table for the relevant fee types and formats the table into a required format
    """
    pmt = data_dict["payments"].copy()
    pmt = create_keys(pmt)
    mask = pmt["amount-description"].isin(req_fee_types)
    filtered_pmt = pmt[mask].copy()
    filtered_pmt["amount"] = filtered_pmt["amount"].fillna(0)
    table = pd.pivot_table(
        filtered_pmt,
        index="key",
        columns="amount-description",
        values="amount",
        aggfunc="sum",
    )
    table = table.reset_index()
    act_cols = [c for c in req_fee_types if c in table.columns]
    table[act_cols] = table[act_cols].fillna(0)
    data_dict["output"] = table
    data_dict["payments"] = pmt

    return data_dict


def get_details_against_keys(
    data_dict: dict,
) -> dict:
    """
    Adds order-id and other information against each key
    """

    table = data_dict["output"].copy()
    order = data_dict["orders"].copy()
    model_mapper = data_dict["fultilment_model_mapper"].copy()
    order = create_keys(order)

    order_date_mapper = order.groupby("key")["purchase-date"].first().to_dict()
    quantity_mapper = order.groupby("key")["quantity"].sum().to_dict()
    table["order-id"] = table["key"].astype(str).str.split("@").str[0]
    table["asin"] = table["key"].astype(str).str.split("@").str[1]
    table["order_date"] = table["key"].map(order_date_mapper)
    table["quantity"] = table["key"].map(quantity_mapper)
    table["model"] = table["order-id"].map(model_mapper)
    data_dict["order"] = order
    data_dict["output"] = table
    data_dict["order_date_mapper"] = order_date_mapper

    return data_dict


def create_referral_category_mapper(data_dict: dict) -> dict:
    """
    This function uses NLP models to determine the billing category of an item based on the product name
    """

    orders = data_dict["orders"]
    asin_mapper = orders.groupby("asin")["product-name"].first().to_dict()

    if "category_mapper" in data_dict:
        category_mapper_asins = list(set(data_dict.get("category_mapper")["asin"]))
        order_asins = list(set(data_dict.get("orders")["asin"]))
        missing_asins = [a for a in order_asins if a not in category_mapper_asins]

        if not missing_asins:
            return data_dict
        else:
            product_names = [asin_mapper.get(a) for a in missing_asins]

    else:
        product_names = list(orders["product-name"].unique())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    commissions_rate_card = data_dict["ratecard"].get("Commissions")
    cats = list(set(commissions_rate_card["Category"]))
    billing_cat = {}

    category_embeddings = model.encode(cats, convert_to_tensor=True)

    for p in product_names:
        product_embedding = model.encode(p, convert_to_tensor=True)
        scores = util.cos_sim(product_embedding, category_embeddings)[0]
        best_idx = scores.argmax().item()
        billing_cat[p] = cats[best_idx]

    category_mapper_new = {k: billing_cat[asin_mapper.get(k)] for k in asin_mapper}

    if "category_mapper" in data_dict:
        category_mapper = data_dict["category_mapper"]
        category_mapper = pd.concat([category_mapper, category_mapper_new], axis=0)
    else:
        category_mapper = category_mapper_new

    df = pd.DataFrame(list(category_mapper.items()), columns=["asin", "category"])
    df["product-name"] = df["asin"].map(asin_mapper)
    df.to_csv("commission_category_mapper.csv", index=False)
    data_dict["category_mapper"] = df

    return data_dict


def extract_rate(fee):
    if fee is None:
        return 0.0

    fee_str = str(fee)
    match = re.search(r"([\d.]+)%", fee_str)
    if match:
        try:
            return float(match.group(1)) / 100
        except ValueError:
            return 0.0


def extract_lower_bound_commission_price_bucket(fee):
    if fee is None:
        return 0

    fee_str = str(fee)
    match = re.search(r">=?\s*([\d.]+)", fee_str)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0


def extract_upper_bound_commission_price_bucket(fee):
    if fee is None:
        return 0

    fee_str = str(fee)
    match = re.search(r"<=?\s*([\d.]+)", fee_str)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0


def clean_rate_card(data_dict: dict) -> dict:
    """
    Calculates commissions by applying relevant price rules to each category
    """

    rate_card = data_dict.get("ratecard")
    commission = rate_card.get("Commissions")
    commission["rate"] = commission["Fee"].apply(extract_rate)
    commission["lower"] = commission["Fee"].apply(
        extract_lower_bound_commission_price_bucket
    )
    commission["upper"] = commission["Fee"].apply(
        extract_upper_bound_commission_price_bucket
    )
    cols = ["rate", "lower"]
    commission[cols] = commission[cols].fillna(0)
    commission["upper"] = commission["upper"].fillna(999999)

    rate_card["Commissions"] = commission
    data_dict["ratecard"] = rate_card

    return data_dict


def calculate_commissions(data_dict: dict) -> dict:
    """
    For each order, get the relevant price bucket
    """

    table = data_dict["output"]
    commissions_rate_card = data_dict.get("ratecard")["Commissions"]
    commissions_rate_card.columns = (
        commissions_rate_card.columns.astype(str).str.lower().str.strip()
    )
    date_cols = ["start date", "end date"]

    for col in date_cols:
        commissions_rate_card[col] = pd.to_datetime(
            commissions_rate_card[col], errors="coerce"
        )

    category_mapper_df = data_dict["category_mapper"]
    category_mapper = category_mapper_df.set_index("asin")["category"].to_dict()

    table["category"] = table["asin"].map(category_mapper)
    table["selling_price"] = table["Principal"] + table["Product Tax"]
    table["selling_price"] = table["selling_price"].astype(float)
    commissions_rate_card["upper"] = commissions_rate_card["upper"].astype(float)
    commissions_rate_card["lower"] = commissions_rate_card["lower"].astype(float)
    commissions_rate_card["rate"] = commissions_rate_card["rate"].astype(float)

    table = table.merge(commissions_rate_card, on="category", how="left")
    mask = (
        (table["selling_price"] > table["lower"])
        & (table["selling_price"] <= table["upper"])
        & (table["order_date"] >= table["start date"])
        & (table["order_date"] <= table["end date"])
    )
    table = table[mask].copy()
    # table.drop(columns=["upper", "lower", "start date", "end date"], inplace=True)
    table["commission_calculated"] = table["selling_price"] * table["rate"]
    commission_dict = table.set_index("key")["commission_calculated"].to_dict()
    upper_dict = table.set_index("key")["upper"].to_dict()
    lower_dict = table.set_index("key")["lower"].to_dict()
    table = data_dict["output"]
    table["commission_calculated"] = table["key"].map(commission_dict)
    table["upper"] = table["key"].map(upper_dict)
    table["lower"] = table["key"].map(lower_dict)
    data_dict["output"] = table

    return data_dict


def force_orders_datetime_format(data_dict: dict) -> dict:
    """ """
    df = data_dict["orders"]
    cols = ["purchase-date"]

    for col in cols:
        aware_mask = df[col].astype(str).str.contains(r"T")
        naive_mask = ~aware_mask
        df.loc[aware_mask, col] = pd.to_datetime(
            df.loc[aware_mask, col], errors="coerce"
        ).dt.tz_convert("Asia/Kolkata")
        df.loc[naive_mask, col] = pd.to_datetime(
            df.loc[naive_mask, col], errors="coerce"
        ).dt.tz_convert("Asia/Kolkata")
        df[col] = (
            pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None).dt.normalize()
        )

    data_dict["orders"] = df

    return data_dict


def calculate_shipping_weights(
    data_dict: dict,
    oms_name: str,
) -> dict:
    """
    Calculates shipping weight as the maximum of dead weight and volumetric weight
    """

    weight_mapper = data_dict["weight_mapper"]
    table = data_dict["output"]
    weight_mapper.columns = weight_mapper.columns.astype(str).str.lower().str.strip()
    req_cols = ["l", "b", "h", "dead_weight"]
    if req_cols not in weight_mapper:
        if oms_name == "EZ":
            weight_mapper["sku"] = (
                weight_mapper["sku"].astype(str).str.split("`").str[1]
            )
            weight_mapper.rename(
                columns={
                    "weight(gm)": "dead_weight",
                    "length(cm)": "l",
                    "width(cm)": "b",
                    "height(cm)": "h",
                },
                inplace=True,
            )

    orders = data_dict["orders"]

    sku_mapper = orders.set_index("sku")["asin"].to_dict()
    weight_mapper["asin"] = weight_mapper["sku"].map(sku_mapper)
    weight_mapper["vol_weight"] = (
        weight_mapper["l"] * weight_mapper["b"] * weight_mapper["h"] / 5
    )
    cols = ["dead_weight", "vol_weight"]
    weight_mapper["shipping_weight"] = weight_mapper[cols].max(axis=1)
    shipping_weight_mapper = weight_mapper.set_index("asin")[
        "shipping_weight"
    ].to_dict()
    table["shipping_weight"] = table["asin"].map(shipping_weight_mapper)
    table["shipping_weight"] = table["shipping_weight"].fillna(500)

    data_dict["output"] = table
    data_dict["shipping_weight_mapper"] = shipping_weight_mapper
    data_dict["weight_mapper"] = weight_mapper

    return data_dict


def assign_weight_buckets(data_dict) -> dict:
    """
    Assigns a weigh bucket based on the caclulated shipping weight.
    Since
    """
    table = data_dict["output"]
    table["shipping_weight_bucket"] = (table["shipping_weight"].fillna(0) / 500).fillna(
        0
    ).astype(int) + 1
    mask = (table["shipping_weight_bucket"] > 2) & (
        table["shipping_weight_bucket"] % 2 == 1
    )
    table.loc[mask, "shipping_weight_bucket"] = (
        table.loc[mask, "shipping_weight_bucket"] + 1
    )

    data_dict["output"] = table

    return data_dict


def map_to_from_pincodes(data_dict) -> dict:
    """
    Estimates Local, regional or national shipping based on to and from pincodes
    """

    mtr = data_dict["mtr"]
    postcode_mapper = data_dict["postcode_mapper"].get("postal_codes")
    table = data_dict["output"]

    mtr.columns = mtr.columns.astype(str).str.lower().str.strip()
    postcode_mapper.columns = (
        postcode_mapper.columns.astype(str).str.strip().str.lower()
    )

    mtr["key"] = mtr["order id"].astype(str) + mtr["asin"].astype(str)

    to_dict = mtr.groupby("key")["ship to postal code"].first().to_dict()
    from_dict = mtr.groupby("key")["ship from postal code"].first().to_dict()
    city_dict = postcode_mapper.groupby("pincode")["reconzo city"].first().to_dict()
    zone_dict = postcode_mapper.groupby("pincode")["amazon zones"].first().to_dict()

    table["from"] = table["key"].map(from_dict)
    table["to"] = table["key"].map(to_dict)
    table["to_city"] = table["to"].map(city_dict)
    table["from_city"] = table["from"].map(city_dict)
    table["to_region"] = table["to"].map(zone_dict)
    table["from_region"] = table["from"].map(zone_dict)

    data_dict["output"] = table

    return data_dict


def map_zones(
    data_dict: dict,
    ixd_flag: bool,
) -> dict:
    """
    Add logic for Local, Regional or National
    """

    table = data_dict["output"]
    mask = (
        (table["from_region"].isna())
        | (table["to_region"].isna())
        | (table["from_city"].isna())
        | (table["to_city"].isna())
    )
    table.loc[mask, "shipping_type"] = "Local"

    if not ixd_flag:
        mask = table["from_region"] == table["to_region"]
        table.loc[mask, "shipping_type"] = "Regional"
        mask = table["to_city"] == table["from_city"]
        table.loc[mask, "shipping_type"] = "Local"
        mask = table["from_region"] != table["to_region"]
        table.loc[mask, "shipping_type"] = "National"

        mask = table["Amazon Easy Ship Charges"] != 0
        table.loc[mask, "shipping_type"] == "All"
    elif ixd_flag:
        table["shipping_type"] == "All"

    data_dict["output"] = table

    return data_dict


def map_size_bands(data_dict: dict, oms_name: str = "EZ") -> dict:
    """
    Maps size band for each SKU
    """

    size_band_mapper = data_dict["size_band_mapper"]
    asin_mapper_dict = data_dict["asin_mapper"]
    table = data_dict["output"]

    if "asin" not in size_band_mapper:
        size_band_mapper.columns = (
            size_band_mapper.columns.astype(str).str.strip().str.lower()
        )
        size_band_mapper["sku"] = (
            size_band_mapper["sku"].astype(str).str.split("`").str[1]
        )
        size_band_mapper["asin"] = size_band_mapper["sku"].map(asin_mapper_dict)

    size_band_mapper_dict = size_band_mapper.set_index("asin")[
        "size band classification"
    ].to_dict()
    table["size_band"] = table["asin"].map(size_band_mapper_dict)
    data_dict["output"] = table
    data_dict["size_band_mapper"] = size_band_mapper_dict

    return data_dict


def calculate_shipping_fee(data_dict: dict, step_level: str, ixd_flag: bool) -> dict:
    """ """

    size_band_mapper = data_dict["size_band_mapper"]
    rate_card = data_dict["ratecard"].get("Weight handling fees")
    rate_card.columns = rate_card.columns.astype(str).str.lower().str.strip()
    table = data_dict["output"]
    size_band = list(table["size_band"].unique())

    def process_table(
        table_subset: pd.DataFrame, model_name: str, rate_card: pd.DataFrame
    ) -> pd.DataFrame:
        conditions = (
            (rate_card["step level"] == step_level)
            & (rate_card["size band"].isin(size_band))
            & (rate_card["model"] == model_name)
        )
        rc = rate_card[conditions].copy()
        rc["weight_bucket"] = (rc["max weight"].fillna(0) / 500).fillna(0).astype(int)
        print(f"Index size of {model_name}: {rc.index.size}")
        table_subset = table_subset.merge(
            rc, left_on="shipping_type", right_on="zone", how="left"
        )
        col_name = model_name + "_shipping_fee_calc"
        table_subset.rename(columns={"rate": col_name}, inplace=True)
        print(f"Index size of {model_name} after merging: {table_subset.index.size}")
        cond = (
            (table_subset["weight_bucket"] == table_subset["shipping_weight_bucket"])
            & (table_subset["order_date"] >= table_subset["start date"])
            & (table_subset["order_date"] <= table_subset["end date"])
            & (table_subset["size_band"] == table_subset["size band"])
        )
        table_subset = table_subset[cond].copy()

        return table_subset

    table["counts"] = table["key"].map(table["key"].value_counts())
    print(f"Counts: {table[['counts', 'key']]}")

    if not ixd_flag:
        es_table = table[table["Amazon Easy Ship Charges"] != 0].copy()
        fba_table = table[table["FBA Weight Handling Fee"] != 0].copy()

        print(f"Easy ship index_size: {es_table.index.size}")
        print(f"FBA index_size: {fba_table.index.size}")

        es_table = process_table(es_table, "Easy Ship", rate_card)
        fba_table = process_table(fba_table, "FBA", rate_card)
        print(f"Easy ship index_size: {es_table.index.size}")
        print(f"FBA index_size: {fba_table.index.size}")
        table = pd.concat([es_table, fba_table], axis=0)
    elif ixd_flag:
        table = process_table(table, "IXD", rate_card)

    # cols_to_drop = ["step level", "size band", "min weight", "max weight", "weight_bucket", "zone", "additional fee", "start date", "end date"]
    cols_to_drop = [
        "min weight",
        "max weight",
        "additional fee",
        "start date",
        "end date",
    ]
    table.drop(columns=cols_to_drop, inplace=True)

    data_dict["output"] = table
    data_dict["ratecard"]["Weight handling fees"] = rate_card

    return data_dict


def map_pick_and_pack_fee(data_dict) -> dict:
    """ """
    table = data_dict["output"]
    rate_card = data_dict["ratecard"].get("Pick & pack fee")
    rate_card.columns = rate_card.columns.astype(str).str.lower().str.strip()
    rate_card["weight_bucket"] = (rate_card["max weight"].fillna(0) / 500).astype(int)

    table = table.merge(
        rate_card, on="weight_bucket", how="left", suffixes=["", "_pnp"]
    )
    print(f"Columns after meger: {table.columns}")
    conds = (
        (table["order_date"] >= table["start date"])
        & (table["order_date"] <= table["end date"])
        & (table["size band"] == table["size band_pnp"])
    )
    table = table[conds].copy()
    table.rename(
        columns={"fees per unit": "pick_and_pack_fee_per_unit_calc"}, inplace=True
    )
    cols = [
        "step level",
        "size band_pnp",
        "start date",
        "end date",
        "description",
        "min weight",
        "max weight",
    ]
    table.drop(columns=cols, inplace=True)
    table["pick_and_pack_fee_calc"] = table["pick_and_pack_fee_per_unit_calc"] * table[
        "quantity"
    ].astype(int)

    data_dict["output"] = table
    data_dict["ratecard"]["Pick & pack fee"] = rate_card

    return data_dict


def create_fixed_closing_category_mapper(
    data_dict: dict,
) -> dict:
    """
    This function uses NLP models to determine the billing category of an item based on the product name
    """

    table = data_dict["output"].copy()
    orders = data_dict["orders"].copy()
    asin_mapper = orders.groupby("asin")["product-name"].first().to_dict()

    if "category_mapper_fixed_closing" in data_dict:
        category_mapper_asins = list(
            set(data_dict.get("category_mapper_fixed_closing")["asin"])
        )
        order_asins = list(set(data_dict.get("output")["asin"]))
        missing_asins = [a for a in order_asins if a not in category_mapper_asins]

        if not missing_asins:
            return data_dict
        else:
            product_names = [asin_mapper.get(a) for a in missing_asins]

    else:
        product_names = list(orders["product-name"].unique())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    fixed_closing_rate_card = data_dict["ratecard"].get("Fixed closing fee - final")
    cats = list(set(fixed_closing_rate_card["Category"]))
    billing_cat = {}

    category_embeddings = model.encode(cats, convert_to_tensor=True)

    for p in product_names:
        product_embedding = model.encode(p, convert_to_tensor=True)
        scores = util.cos_sim(product_embedding, category_embeddings)[0]
        best_idx = scores.argmax().item()
        billing_cat[p] = cats[best_idx]

    category_mapper_new = {k: billing_cat[asin_mapper.get(k)] for k in asin_mapper}

    if "category_mapper_fixed_closing" in data_dict:
        category_mapper_fixed_closing = data_dict["category_mapper_fixed_closing"]
        category_mapper_fixed_closing = pd.concat(
            [category_mapper_fixed_closing, category_mapper_new], axis=0
        )
    else:
        category_mapper_fixed_closing = category_mapper_new

    df = pd.DataFrame(
        list(category_mapper_fixed_closing.items()),
        columns=["asin", "category_fixed_closing"],
    )
    df["product-name"] = df["asin"].map(asin_mapper)
    df.to_csv("fixed_closing_category_mapper.csv", index=False)
    data_dict["category_mapper_fixed_closing"] = df

    return data_dict


def get_fixed_closing_fees_categories(data_dict) -> dict:
    """
    Placeholder function - does noting right now
    """

    table = data_dict["output"].copy()
    category_mapper = data_dict["category_mapper_fixed_closing"].copy()
    category_mapper.columns = (
        category_mapper.columns.astype(str).str.lower().str.strip()
    )
    category_mapper.drop(columns=["product-name"], inplace=True)

    table = table.merge(category_mapper, on="asin", how="left")

    data_dict["output"] = table

    return data_dict


def calculate_fixed_closing_fees(data_dict: dict) -> dict:
    """ """
    rate_card = data_dict.get("ratecard").get("Fixed closing fee - final").copy()
    rate_card.columns = rate_card.columns.astype(str).str.lower().str.strip()
    table = data_dict["output"].copy()

    table = table.merge(
        rate_card,
        left_on="category_fixed_closing",
        right_on="category",
        how="left",
        suffixes=["", "_fc"],
    )

    conds = (
        (table["order_date"] >= table["start"])
        & (table["order_date"] <= table["end"])
        & (table["order_type"] == table["model_fc"])
        & (table["selling_price"] >= table["min price"])
        & (table["selling_price"] <= table["max price"])
    )
    table = table[conds].copy()

    cols = ["start", "end", "model_fc", "min price", "max price"]
    table.drop(columns=cols, inplace=True)
    data_dict["output"] = table

    return data_dict


def complete_fee_audit_amazon(
    _path_: str = "/Users/ap/Desktop/P-Tal/Amazon India Fee Audit",
    req_fee_types: list = [
        "Principal",
        "Product Tax",
        "FBA Weight Handling Fee",
        "Fixed closing fee",
        "FBA Pick & Pack Fee",
        "Commission",
        "Technology Fee",
        "TechnologyFee",
        "Refund commission",
        "RemovalComplete",
        "Amazon Easy Ship Charges",
        "Storage",
    ],
    oms_name: str = "EZ",
    step_level: str = "Standard",
    ixd_flag: bool = False,
) -> dict:
    """
        Complete fee audit for Amazon. This required the following files:

    ||------------------------------|------------------------------|------------------------------||
    || File required                | Naming convention            | Remarks                      ||
    ||------------------------------|------------------------------|------------------------------||
    || Payments file                | agg_payments                 | As downloaded from the API   ||
    ||------------------------------|------------------------------|------------------------------||
    || orders                       | agg_orders                   | As downloaded from the API   ||
    ||------------------------------|------------------------------|------------------------------||
    || returns                      | agg_returns                  | As downloaded from the API   ||
    ||------------------------------|------------------------------|------------------------------||
    || mfn_returns                  | agg_mfn-returns              | As downloaded from the API   ||
    ||------------------------------|------------------------------|------------------------------||
    || replacements                 | agg_replacements             | As downloaded from the API   ||
    ||------------------------------|------------------------------|------------------------------||
    || shipment                     | agg_shipments                | As downloaded from the API   ||
    ||------------------------------|------------------------------|------------------------------||
    || Updated rate card            | rate_card                    | Prepared by Reconzo          ||
    ||------------------------------|------------------------------|------------------------------||
    || MTR                          | mtr                          | Downloaded from seller       ||
    ||                              |                              | central website.             ||
    ||------------------------------|------------------------------|------------------------------||
    || Weight mapper                | weight_mapper                | This is derived from WMS or  ||
    ||                              |                              | fed manually. Should contain ||
    ||                              |                              | 'l', 'b', 'h' and            ||
    ||                              |                              | 'dead_weight' columns        ||
    ||------------------------------|------------------------------|------------------------------||
    || postcode mapper              | postcode                     | Prepared by Reconzo          ||
    ||------------------------------|------------------------------|------------------------------||
    || fulfilment model mapper      | fulfilment_model_mapper      | from client. Must contain    ||
    ||                              |                              | 'order-id' and 'model' fileds||
    ||------------------------------|------------------------------|------------------------------||
    || size band mapper             | size_band_mapper             | Should contain "asin" and    ||
    ||                              |                              | "size_band_classification"   ||
    ||                              |                              | columns                      ||
    ||------------------------------|------------------------------|------------------------------||
    || commission_category mapper   | commission_category_mapper   | The code prepares a category ||
    || [OPTIONAL]                   |                              | mapper. However, it needs    ||
    ||                              |                              | refinement. File is stored   ||
    ||                              |                              | in the same location. Rerun  ||
    ||                              |                              | the code after making        ||
    ||                              |                              | refinements.                 ||
    ||------------------------------|------------------------------|------------------------------||
    || fixed closing category mapper| fixed_closing_category_mapper| The code prepares a category ||
    || [OPTIONAL]                   |                              | mapper. However, it needs    ||
    ||                              |                              | refinement. File is stored   ||
    ||                              |                              | in the same location. Rerun  ||
    ||                              |                              | the code after making        ||
    ||                              |                              | refinements.                 ||
    ||------------------------------|------------------------------|------------------------------||


    Parameters:
    -----------
        _path_: string
                The path to where all the above files reside

        req_fee_types: list
                List of fee types to audit. Default is ["Principal", "Product Tax", "FBA Weight Handling Fee",
                "Fixed closing fee", "FBA Pick & Pack Fee", "Commission", "Technology Fee", "TechnologyFee",
                "Refund commission", "RemovalComplete", "Amazon Easy Ship Charges", "Storage"]

        oms_name: string
                Name of the OMS used ("EZ" for EasyEcom and "UC for Unicommerce). Default is "EZ".

        step_level: string
                STEP Level of the client. Default is "Standard". First letter needs to be Capitalized
                valid entries are ["Premium", "Advanced", "Standard", "Basic"]

        ixd_flag: Boolean
                Whether the client is using IXD for fulfilment. Default is False

        Usage:
        ------
        data_dict = complete_fee_audit_amazon(_path_=path, req_fee_types=list, oms_name="EZ", step_level="Basic", ixd_flag=True)

        Output:
        -------
        data_dict: Dictionary
                A dictionary containing the fee audit. Final file is stored in data_dict["output"]
                Feel free to explotre other key-value pairs within the dicitonary

    """

    print(f"Complete fee audit for Amazon. This required the following files:")
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| File required                | Naming convention            | Remarks                      ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| Payments file                | agg_payments                 | As downloaded from the API   ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| orders file                  | agg_orders                   | As downloaded from the API   ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| returns                      | agg_returns                  | As downloaded from the API   ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| mfn returns                  | agg_mfn-returns              | As downloaded from the API   ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| replacements                 | agg_replacements             | As downloaded from the API   ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| shipments                    | agg_shipments                | As downloaded from the API   ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| Updated rate card            | rate_card                    | Prepared by Reconzo          ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| MTR                          | mtr                          | Downloaded from seller       ||"
    )
    print(
        f"||                              |                              | central website.             ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| Weight mapper                | weight_mapper                | This is derived from WMS or  ||"
    )
    print(
        f"||                              |                              | fed manually. Should contain ||"
    )
    print(
        f"||                              |                              | 'l', 'b', 'h' and            ||"
    )
    print(
        f"||                              |                              | 'dead_weight' columns        ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| postcode mapper              | postcode                     | Prepared by Reconzo          ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| fulfilment model mapper      | fulfilment_model_mapper      | from client, must contain    ||"
    )
    print(
        f"||                              |                              | 'order-id' and 'model' fields||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| size band mapper             | size_band_mapper             | Should contain 'asin' and    ||"
    )
    print(
        f"||                              |                              | 'size_band_classification'   ||"
    )
    print(
        f"||                              |                              | columns                      ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| commission_category mapper   | commission_category_mapper   | The code prepares a category ||"
    )
    print(
        f"|| [OPTIONAL]                   |                              | mapper. However, it needs    ||"
    )
    print(
        f"||                              |                              | refinement. File is stored   ||"
    )
    print(
        f"||                              |                              | in the same location. Rerun  ||"
    )
    print(
        f"||                              |                              | the code after making        ||"
    )
    print(
        f"||                              |                              | refinements.                 ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )
    print(
        f"|| fixed closing category mapper| fixed_closing_category_mapper| The code prepares a category ||"
    )
    print(
        f"|| [OPTIONAL]                   |                              | mapper. However, it needs    ||"
    )
    print(
        f"||                              |                              | refinement. File is stored   ||"
    )
    print(
        f"||                              |                              | in the same location. Rerun  ||"
    )
    print(
        f"||                              |                              | the code after making        ||"
    )
    print(
        f"||                              |                              | refinements.                 ||"
    )
    print(
        f"||------------------------------|------------------------------|------------------------------||"
    )

    valid_step_levels = ["Premium", "Advanced", "Standard", "Basic"]
    if step_level not in valid_step_levels:
        msg = f"{step_level} not in {valid_step_levels}. Please correct the input"
        warnings.warn(msg)
        sys.exit(1)

    integrated_oms = ["UC", "EZ"]
    if oms_name not in integrated_oms:
        msg = f"{oms_name} not integrated yet!"
        warnings.warn(msg)
        sys.exit(1)

    data_dict = read_files(_path_=_path_)
    data_dict = force_orders_datetime_format(data_dict)
    data_dict = rename_cols(data_dict)
    data_dict = add_asin_to_pmt(data_dict)
    data_dict = forward_fill_settlement_date(
        data_dict,
        cols_to_ffill=["settlement-start-date", "settlement-end-date", "deposit-date"],
    )
    data_dict = create_fee_table(data_dict=data_dict, req_fee_types=req_fee_types)
    data_dict = get_details_against_keys(data_dict)
    data_dict = create_referral_category_mapper(data_dict)
    data_dict = clean_rate_card(data_dict)
    data_dict = calculate_commissions(data_dict)
    data_dict = calculate_shipping_weights(data_dict, oms_name)
    data_dict = assign_weight_buckets(data_dict)
    data_dict = map_to_from_pincodes(data_dict)
    data_dict = map_zones(data_dict, ixd_flag)

    ## PLACEHOLDER - TO Be DELETED
    random_mapper = {"1": "Local", "2": "Regional", "3": "National"}
    random_mapper2 = {"1": "FBA", "2": "Easy Ship", "3": "Seller Flex"}
    table = data_dict["output"]
    table["rnd"] = np.random.randint(1, 4, size=len(table))
    table["rnd"] = table["rnd"].astype(str)
    table["shipping_type"] = table["rnd"].map(random_mapper)
    table["order_type"] = table["rnd"].map(random_mapper2)
    table.drop(columns=["rnd"], inplace=True)
    data_dict["output"] = table

    data_dict = map_size_bands(data_dict, oms_name="EZ")
    data_dict = calculate_shipping_fee(data_dict, step_level, ixd_flag)
    print(f"After merging size of the dataframe: {data_dict['output'].index.size}")
    data_dict = map_pick_and_pack_fee(data_dict)
    print(f"After merging size of the dataframe: {data_dict['output'].index.size}")
    data_dict = create_fixed_closing_category_mapper(data_dict)
    data_dict = get_fixed_closing_fees_categories(data_dict)
    data_dict = calculate_fixed_closing_fees(data_dict)

    return data_dict
