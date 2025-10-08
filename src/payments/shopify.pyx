import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
from typing import Optional
import sys
import time


def change_dict_keys_to_lower_case(some_dict: dict) -> dict:
    some_dict = {k.lower(): v for k, v in some_dict.items()}
    return some_dict


def standardize_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    This funciton renames columns and standardizes columns and values for tshopify orders
    """

    orders_rename_dict = {
        "name": "order_id",
        "lineitem sku": "sku",
        "lineitem price": "unit_price",
        "discount amount": "discount",
        "lineitem quantity": "quantity_ordered",
        "taxes": "taxes",
        "shipping": "shipping",
        "shipping method": "shopify_payment_method",
        "payment reference": "pg_reference",
        "refunded amount": "refunded_amount",
    }

    df = df[orders_rename_dict.keys()].rename(columns=orders_rename_dict)
    mask_non_sku = df["sku"].isna()
    df = df[~mask_non_sku].copy()
    mask = df["order_id"].str.contains("#")
    df.loc[mask, "order_id"] = (
        df.loc[mask, "order_id"].astype(str).str.split("#").str[1]
    )
    agg_func = {
        k: (
            "first"
            if k
            in [
                "order_id",
                "sku",
                "shopify_payment_method",
                "pg_reference",
                "unit_price",
            ]
            else "sum"
        )
        for k in df.columns
    }
    df["key"] = df["order_id"].astype(str) + "-" + df["sku"].astype(str)
    df = df.groupby("key", as_index=False).agg(agg_func)
    df["gross_sales"] = df["unit_price"] * df["quantity_ordered"]
    df["sku"] = df["sku"].astype(str)
    df["order_id"] = df["order_id"].astype(str)

    return df


def add_payment_reference_number_for_each_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    By default, shopify adds payment reference number against only 1 order ID. We need to fill empty rows with the payment reference number
    Arguments:
    a. order_df: Shopify order_export file
    """

    mask = df["pg_reference"].isna()
    pg_ref_dict = df[~mask].set_index("order_id")["pg_reference"].to_dict()
    df["pg_reference"] = df["order_id"].map(pg_ref_dict)

    return df


def calculate_order_pct_allocation(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the allocation ratio for each sub-order within the order
    Arguments:
    a. df: A shopify order_export dataframe
    """

    total_order_amount_dict = df.groupby("order_id")["gross_sales"].sum()
    df = df.assign(
        total_amount=lambda x: x["order_id"].map(total_order_amount_dict),
        reciprocal_tot_amount=lambda x: 1 / x["total_amount"],
        order_pct_allocation=lambda x: x["gross_sales"] * x["reciprocal_tot_amount"],
    ).drop(columns=["total_amount", "reciprocal_tot_amount"])

    return df


def allocate_discounts_to_suborder(df: pd.DataFrame) -> pd.DataFrame:
    """
    By default, shopify adds discount to only the first instance of the order_id. We will allocate the discount amount to each sub-order within the order_id
    Arguments:
    a. df: Shopify order_exports file
    """

    cols = ["order_pct_allocation"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warn_str = f"{missing} not found in DataFrame. Run 'calculate_order_pct_allocation' before calling this funciton."
        warnings.warn(warn_str)
        sys.exit(1)

    total_order_discount_dict = df.groupby("order_id")["discount"].sum()
    df = df.assign(
        tot_discount=lambda x: x["order_id"].map(total_order_discount_dict),
        disc_allocated=lambda x: x["order_pct_allocation"] * x["tot_discount"],
    ).drop(columns=["tot_discount"])

    return df


def allocate_shipping_to_suborder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocates shipping charge to each sub-order. This is required because shopify charges shipping charges at an orderID level and is only charged to the first instance
    since we are aggregating at a "key" level, this charge will be missed (sku against shipping charge is NaN)
    Arguments:
    a. df: order dataframe, which includes order_pct_allocation columns. So this function can only be called after "allocate_dicscounts_to_suborder" function has run
    """

    cols = ["order_pct_allocation"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warn_str = f"{missing} not found in DataFrame. Run 'calculate_order_pct_allocation' before calling this funciton."
        warnings.warn(warn_str)
        sys.exit(1)

    total_shipping_dict = df.groupby("order_id")["shipping"].sum()
    df = df.assign(
        tot_shipping_charges=lambda x: x["order_id"].map(total_shipping_dict),
        shipping_allocated=lambda x: x["tot_shipping_charges"]
        * x["order_pct_allocation"],
    ).drop(columns=["tot_shipping_charges"])

    return df


def allocate_refunds_to_suborder(df: pd.DataFrame) -> pd.DataFrame:
    """
    By default, shopify adds refunds to only the first instance of the order_id. We will allocate the refund amount to each sub-order within the order_id
    Arguments:
    a. df: Shopify order_exports file
    """

    cols = ["order_pct_allocation"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warn_str = f"{missing} not found in DataFrame. Run 'calculate_order_pct_allocation' before calling this funciton."
        warnings.warn(warn_str)
        sys.exit(1)

    total_order_refund_dict = df.groupby("order_id")["refunded_amount"].sum()
    df = df.assign(
        tot_refunds=lambda x: x["order_id"].map(total_order_refund_dict),
        refund_allocated=lambda x: x["order_pct_allocation"] * x["tot_refunds"],
    ).drop(columns=["tot_refunds"])

    return df


def calculate_amount_to_be_collected(df: pd.DataFrame) -> pd.DataFrame:

    df["gross_sales_incl_shipping"] = df["gross_sales"] + df["shipping"]
    df["selling_price"] = df["gross_sales"] - df["disc_allocated"]
    df["amount_to_be_collected"] = (
        df["selling_price"] + df["shipping_allocated"] - df["refund_allocated"]
    )
    df["net_sales"] = (
        df["selling_price"]
        - df["taxes"]
        + df["shipping_allocated"]
        - df["refund_allocated"]
    )

    return df


def identify_oms(df: pd.DataFrame) -> str:
    if "ee invoice no" in df.columns.astype(str).str.lower().str.strip():
        return "EZ"
    else:
        return "UC"


def clean_oms(df: pd.DataFrame, oms_name: str) -> pd.DataFrame:
    """
    Clean certain columns with backtick "`" in the value
    Arguments:
    1. df: OMS dataframe
    2. oms_name: Name of the OMS
    """

    if oms_name == "EZ":
        mask = df["awb no"].astype(str).str.contains("`")
        df.loc[mask, "awb no"] = (
            df.loc[mask, "awb no"].astype(str).str.split("`").str[1]
        )
        # df['reference code'] = df['reference code'].astype(str).str.split("`").str[1]

    return df


def get_required_oms_columns(oms_name: str) -> dict:
    """
    This function takes the name of the oms software and returns a list of required column names for each. Currently built for Unicommerce and EasyEcom only.
    To add new oms - add them to identify_oms function and against the string, add relevant column names here.
    Structure of the columns is as follows:
    1. shopify_order_id
    2. suborder_number
    3. order_status
    4. shipping_status
    5. sku
    6. marketplace_sku
    7. product_name
    8. brand
    9. payment_method
    10. order_date
    11. Quantity [Note: Quantity is forced = 1 for Unicommerce]
    12. awb no
    """

    oms_cols_dict = {
        "EZ": [
            "reference code",
            "suborder no",
            "order status",
            "shipping status",
            "sku",
            "marketplace sku",
            "product name",
            "brand",
            "payment mode",
            "order date",
            "item quantity",
            "awb no",
        ],
        "UC": [
            "display order code",
            "sale order item code",
            "sale order item status",
            "shipping courier status",
            "item sku code",
            "seller sku code",
            "sku name",
            "item type brand",
            "cod",
            "order date as dd/mm/yyyy hh:mm:ss",
            "tracking number",
        ],
    }

    std_col_names = {
        "EZ": [
            "order_id",
            "suborder_id",
            "order_status",
            "shipping_status",
            "sku",
            "marketplace_sku",
            "product_name",
            "brand",
            "payment_method",
            "order_date",
            "quantity_oms",
            "awbno",
        ],
        "UC": [
            "order_id",
            "suborder_id",
            "order_status",
            "shipping_status",
            "sku",
            "marketplace_sku",
            "product_name",
            "brand",
            "payment_method",
            "order_date",
            "awbno",
        ],
    }

    oms_rename_dict = {
        k: v for k, v in zip(oms_cols_dict.get(oms_name), std_col_names.get(oms_name))
    }

    return oms_rename_dict


def standardize_oms(
    df: pd.DataFrame, oms_name: str, oms_rename_dict: dict
) -> pd.DataFrame():
    """
    This function standardizes oms dataframe
    """
    df = df[oms_rename_dict.keys()].rename(columns=oms_rename_dict)
    df["order_status"] = df["order_status"].astype(str).str.lower().str.strip()
    df["shipping_status"] = df["shipping_status"].astype(str).str.lower().str.strip()

    if oms_name == "UC":
        df["payment_method"] = df["payment_method"].fillna(0)
        mask = df["payment_method"] == 1
        df.loc[mask, "payment_method"] = "COD"
        mask = df["payment_method"] == 0
        df.loc[mask, "payment_method"] = "PrePaid"
        df["quantity_oms"] = 1
        df["order_date"] = pd.to_datetime(
            df["order_date"], errors="coerce", format="%d/%m/%y %H:%M"
        )
    elif oms_name == "EZ":
        # df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", format="%d/%m/%Y %H:%M:%S")
        df["order_date"] = pd.to_datetime(
            df["order_date"], errors="coerce", dayfirst=True
        )

    df["month"] = df["order_date"].dt.to_period("M")
    df["order_date"] = df["order_date"].dt.date
    df["month"] = df["month"].astype(str)
    time.sleep(1)

    return df


def clean_logistics(
    logistics_dict: dict, logistics_providers: list, oms_df: pd.DataFrame
) -> dict:
    """
    This function will clean the waybill numbers for delhivery logsitics invoices
    Arguments:
    a. logistics_dict: A dictionary of all logistics cost charged by various 3pl's. Read the excel file as a dictionary by giving sheet_name=None parameter
    """

    logistics_dict = change_dict_keys_to_lower_case(logistics_dict)
    oms_for_mapper = oms_df[~oms_df["awbno"].isna()].copy()
    oms_for_mapper["awbno"] = oms_for_mapper["awbno"].astype(str)
    awb_mapper = oms_for_mapper.groupby("awbno")["order_id"].first().to_dict()

    for provider in logistics_dict:
        df = logistics_dict.get(provider)
        if provider == "delhivery":
            # df['waybill_num'] = df['waybill_num'].astype(str).str.split('="').str[1]
            # df['waybill_num'] = df['waybill_num'].astype(str).str.split('"').str[0]
            df["pickup_pincode"] = pd.NA
            df["rto_date"] = pd.NA
            df["courier_code"] = "DEL"
            logistics_dict[provider] = df
        elif provider == "bluedart":
            df["order_id"] = (
                df["CAWBNO"].astype(str).str.lower().str.strip().map(awb_mapper)
            )
            df["status"] = pd.NA
            df["zone"] = pd.NA
            df["cgst"] = 0
            df["sgst"] = 0
            df["igst"] = (
                df["NTOTAMOUNT"] * 0.18
            )  ### DEBUG: Need to confirm if the amount is before or after GST
            df["gross_total_logistics"] = (
                df["NTOTAMOUNT"] * 1.18
            )  ### DEBUG: Need to confirm if the amount is before or after GST
            df["delivery_date"] = pd.NaT
            df["rto_date"] = pd.NaT
            df["courier_code"] = "BLDRT"
            logistics_dict[provider] = df
        elif provider == "dtdc":
            df["order_id"] = (
                df["Awb No"].astype(str).str.lower().str.strip().map(awb_mapper)
            )
            df["status"] = pd.NA
            df["from"] = pd.NA
            df["zone"] = pd.NA
            cost_cols = ["Fov Amount", "Docket", "Other", "Freight", "Fuel"]
            df["net_logistics_cost"] = df[cost_cols].sum(axis=1)
            df["cgst"] = 0
            df["sgst"] = 0
            df["delivery_date"] = pd.NaT
            df["rto_date"] = pd.NaT
            df["courier_code"] = "DTDC"
            logistics_dict[provider] = df

    return logistics_dict, awb_mapper


def standardize_logistics(logistics_dict: dict) -> pd.DataFrame:
    """
    This function standardizes logistics cost by different 3pl partners
    Arguments:
    a. logistics_dict: A dictionary of all logistics cost charged by various 3pl's. Read the excel file as a dictionary by giving sheet_name=None parameter
    """

    logistics_dict = change_dict_keys_to_lower_case(logistics_dict)

    rename_dict = {
        "shipdelight": [
            "airwaybilno",
            "orderno",
            "actual weight",
            "pickup date",
            "status",
            "pickup pincode",
            "delivery pincode",
            "zone",
            "sub total",
            "cgst",
            "sgst",
            "igst",
            "total freight",
            "delivery date",
            "rto date",
            "courier code",
        ],
        "delhivery": [
            "waybill_num",
            "order_id",
            "charged_weight",
            "pickup_date",
            "status",
            "pickup_pincode",
            "destination_pin",
            "zone",
            "gross_amount",
            "cgst",
            "sgst/ugst",
            "igst",
            "total_amount",
            "fpd",
            "rto_date",
            "courier_code",
        ],
        "bluedart": [
            "cawbno",
            "order_id",
            "nactwgt",
            "dbatchdt",
            "status",
            "corgarea",
            "destpincode",
            "zone",
            "ntotamount",
            "cgst",
            "sgst",
            "igst",
            "gross_total_logistics",
            "delivery_date",
            "rto_date",
            "courier_code",
        ],
        "dtdc": [
            "awb no",
            "order_id",
            "bill wt",
            "date",
            "status",
            "from",
            "destination",
            "zone",
            "net_logistics_cost",
            "cgst",
            "sgst",
            "tax",
            "total",
            "date",
            "rto_date",
            "courier_code",
        ],
    }

    logistics_standard_col_names = [
        "awb",
        "order_id",
        "act_wt",
        "pickup_date",
        "status",
        "from_pincode",
        "to_pincode",
        "zone",
        "net_logistics_cost",
        "cgst",
        "sgst",
        "igst",
        "gross_total_logistics",
        "delivery_date",
        "rto_date",
        "courier_code",
    ]

    missing_logistics = [x for x in logistics_dict if x not in rename_dict]
    if missing_logistics:
        warn_str = f"{missing_logistics} columns not found in logistics stand. Please write code to standardize them"
        warnings.warn(warn_str)
        sys.exit(1)

    list_of_dfs = []

    for provider, temp_df in logistics_dict.items():
        req_cols = rename_dict.get(provider)
        req_cols = [x.lower() for x in req_cols]
        temp_df.columns = temp_df.columns.astype(str).str.lower().str.strip()
        temp_df = temp_df[rename_dict.get(provider)]
        time.sleep(0.2)
        temp_df.columns = logistics_standard_col_names
        temp_df["3pl_logistics_name"] = provider
        num_cols = [
            "act_wt",
            "net_logistics_cost",
            "cgst",
            "sgst",
            "igst",
            "gross_total_logistics",
        ]
        agg_func = {
            k: ("sum" if k in num_cols else "first")
            for k in temp_df.columns
            if k != "order_id"
        }
        temp_df = temp_df.groupby("order_id", as_index=False).agg(agg_func)
        list_of_dfs.append(temp_df)

    df = pd.concat(list_of_dfs, axis=0)
    mask = df["order_id"].astype(str).str.contains("#")
    df.loc[mask, "order_id"] = (
        df.loc[mask, "order_id"].astype(str).str.split("#").str[1]
    )
    df["order_id"] = df["order_id"].astype(str)
    mask = df["order_id"].astype(str).str.contains(".")
    df["order_id"] = df["order_id"].astype(str)
    df.loc[mask, "order_id"] = (
        df.loc[mask, "order_id"].astype(str).str.split(".").str[0]
    )
    df["order_id"] = df["order_id"].astype(str)

    return df, logistics_standard_col_names


def clean_cogs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cogs file to get monthwise applicable cogs
    Arguments:
    a. df: "Additional Data" sheet by Reconzo
    """

    df.columns = df.columns.astype(str).str.lower().str.strip()
    df["channel"] = df["channel"].fillna("all")
    df["channel"] = df["channel"].astype(str).str.lower().str.strip()
    rename_dict = {
        "sku - internal": "sku",
        "applicable month": "date",
        "channel": "channel",
        "cogs": "cogs_per_unit",
    }
    df = df[rename_dict.keys()].rename(columns=rename_dict)

    df["month"] = pd.to_datetime(
        df["date"], errors="coerce", dayfirst=True
    ).dt.to_period("M")
    df["month"] = df["month"].astype(str)
    df["sku"] = df["sku"].astype(str)
    applicable_channels = list(set(df["channel"]))

    if "shopify" in applicable_channels:
        mask = df["channel"] == "shopify"
    elif "all" in applicable_channels:
        mask = df["channel"] == "all"

    df = df[mask].copy()
    df.drop(columns=["channel"], inplace=True)

    return df


def assign_shipping_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function assigns a status to each sub-order based on shipping status column of the oms.
    Arguments:
    df: Standardized oms dataframe
    """

    unique_ship_status = [
        "reverse pickup scheduled",
        "in transit",
        "delivered",
        "out for delivery",
        "shipment lost",
        "rto in-transit",
        "shipment error",
        "shipment created",
        "rto undelivered",
        "returned",
        "pickup done",
        "courier_return-confirmed",
        "undelivered - consignee refused",
        "courier_return-rto_delivered",
        "rto in-transit",
        "courier_return-delivered",
        "out for delivery",
        "pickedup",
        "courier_return-in_transit",
        "picked",
        "courier_return-out_for_delivery",
        "delivered",
        "rto delivered",
        "courier_return-created",
        "courier_return-rto-delivered",
        "shipped",
        "courier_return-cancelled",
        "pickup cancelled",
        "in-transit",
        "rto undelivered",
        "in_transit",
        "intransit",
        "courier_return-rto-intransit",
        "sd-reattempt",
        "undelivered",
    ]

    ## Create a dictionary for item shipped status first
    status_mapper = {}
    for status in unique_ship_status:
        if "delivered" in status:
            status_mapper[status] = "delivered"
        elif "return" in status:
            status_mapper[status] = "returned"
        elif "rto" in status:
            status_mapper[status] = "rto"
        elif "undelivered" in status:
            status_mapper[status] = "rto"
        elif "reverse" in status:
            status_mapper[status] = "returned"
        elif "cancelled" in status:
            status_mapper[status] = "cancelled"
        elif "return-cancelled" in status:
            status_mapper[status] = "delivered"
        elif "unknown" in status:
            status_mapper[status] = "unknown"
        else:
            status_mapper[status] = "pending/in-transit/on-hold"

    mask = ~df["shipping_status"].isna()
    df.loc[mask, "final_status_sh"] = df.loc[mask, "shipping_status"].map(status_mapper)
    df["final_status_sh"] = df["final_status_sh"].fillna("unknown")

    return df


def assign_order_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns status to each order based on order status column of the oms
    Arguments:
    a. df: standardized oms dataframe.
    """

    unique_order_status = [
        "assigned",
        "cancelled",
        "confirmed",
        "manifest scnanned",
        "on hold",
        "pending",
        "printing",
        "returned",
        "shipped",
        "delivered",
        "dispatched",
        "fulfillable",
    ]
    status_mapper = {}
    for s in unique_order_status:
        if "cancelled" in s:
            status_mapper[s] = "cancelled"
        elif "return" in s:
            status_mapper[s] = "returned"
        elif "rto" in s:
            status_mapper[s] = "rto"
        elif "delivered" in s:
            status_mapper[s] = "delivered"
        else:
            status_mapper[s] = "pending/in-transit/on-hold"

    mask = ~df["order_status"].isna()
    df.loc[mask, "final_status_od"] = df.loc[mask, "order_status"].map(status_mapper)
    df["final_status_od"] = df["final_status_od"].fillna("unknown")

    return df


def assign_final_status_as_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns final status to each order.
    Arguments:
    a. df: standardized oms file, after applying final_status based on order_status and shipping_status
    """

    df["final_status_list"] = df["final_status_sh"]
    mask = df["final_status_list"] == "unknown"
    df.loc[mask, "final_status_list"] = df.loc[mask, "final_status_od"]

    if "unknown" in df["final_status_list"]:
        warn_str = f"\n{'*' * 170}\nUnknown statuses found in oms. Please check and correct the status functions!!\n{'*' * 170}\n"
        warnings.warn(warn_str)

    return df


def aggregate_oms(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function aggregates oms dataframe. Aggregation is custom. We need to aggregate by order_id and sku. Need to capture status of partially delivered orders
    For ex. order_id x sku has 20 units, 15 of which were delivered and 5 were cancelled, we need to get the status & qty as lists for each combo of OID x SKU x final_status.
    So this will be final_status = [delivere, cancelled], quantity_oms=[15,5] for each combo of OID x SKU
    Then after merging this with order file, we can get status of each sub-order, by exploding the merged dataframe on this column.
    Arguments:
    (a) df: Standard oms dataframe. Currently solved for Unicommerce (UC) and EasyEcom (EZ)
    """

    ## Need to aggregate by order_id, sku, and statuses, so that we get partially delivered and sub-order level status information
    mask = df["order_id"].astype(str).str.contains("#")
    df.loc[mask, "order_id"] = (
        df.loc[mask, "order_id"].astype(str).str.split("#").str[1]
    )

    df["key1"] = (
        df["order_id"].astype(str)
        + "-"
        + df["sku"].astype(str)
        + "-"
        + df["final_status_list"].astype(str)
    )
    agg_func = {k: ("first" if k != "quantity_oms" else "sum") for k in df.columns}
    df = df.groupby("key1", as_index=False).agg(agg_func)
    df.drop(columns=["key1"], inplace=True)

    ## Re-aggregate by order_id and sku key to get the final dataframe that can be merged with the order dataframe
    df["key"] = df["order_id"].astype(str) + "-" + df["sku"].astype(str)

    ## Create a new aggregation function, which holds information of order_id and sku key and also stores information of item delivered status as separate lists
    agg_func = {
        k: ("first" if k not in ["final_status_list", "quantity_oms"] else list)
        for k in df.columns
    }
    df = df.groupby("key", as_index=False).agg(agg_func)

    return df


def assign_final_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the partially delivered status for each sub-order / order
    Arguments:
    a. df: dataframe containing processed oms
    """

    mask = df["final_status_list"] == "no_matches"
    no_match_df = df[mask].copy()
    df = df[~mask].copy()

    unique_statuses = {frozenset(x) for x in df["final_status_list"]}

    mapper = {}

    for s in unique_statuses:
        if "delivered" in s and any(
            x in s
            for x in ["returned", "rto", "cancelled", "pending/in-transit/on-hold"]
        ):
            mapper[s] = "partially delivered"
        elif "delivered" in s:
            mapper[s] = "delivered"
        elif any(x in s for x in ["returned", "rto"]):
            mapper[s] = "return/rto"
        elif "cancelled" in s:
            mapper[s] = "cancelled"
        elif "pending/in-transit/on-hold" in s:
            mapper[s] = "pending/in-transit/on-hold"
        else:
            mapper[s] = "unknown"

    df["final_status"] = df["final_status_list"].map(
        lambda x: mapper.get(frozenset(x), "unknown")
    )
    df = pd.concat([df, no_match_df], axis=0, ignore_index=True)

    return df


def calculate_net_delivered_qty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocates net_delivered_quantity based on the final status of the order
    Arguments:
    a. df: a dataframe containing orders file mapped with the oms file
    """

    df = df.reset_index(drop=True)
    df["new_index"] = df.index

    ## Map partially delivered orders
    exploded = df.explode(["final_status_list", "quantity_oms"])

    gross_qty_dict = exploded.groupby("new_index")["quantity_oms"].sum().to_dict()

    cancel_qty = (
        exploded[exploded["final_status_list"] == "cancelled"]
        .groupby("new_index")["quantity_oms"]
        .sum()
        .to_dict()
    )

    return_qty = (
        exploded[exploded["final_status_list"].isin(["returned", "rto"])]
        .groupby("new_index")["quantity_oms"]
        .sum()
        .to_dict()
    )

    mask = exploded["final_status_list"].isin(["returned", "cancelled", "rto"])
    exploded.loc[mask, "quantity_oms"] *= -1
    net_delivered_qty_dict = (
        exploded.groupby("new_index")["quantity_oms"].sum().to_dict()
    )
    df["net_delivered_qty"] = df["new_index"].map(net_delivered_qty_dict).fillna(0)
    df["gross_qty"] = df["new_index"].map(gross_qty_dict).fillna(0)
    df["cancelled_qty"] = df["new_index"].map(cancel_qty).fillna(0)
    df["return_qty"] = df["new_index"].map(return_qty).fillna(0)

    df.drop(columns=["new_index"], inplace=True)

    ## Map returned and cancelled orders as 0
    mask = df["final_status"].isin(["return/rto", "cancelled"])
    df.loc[mask, "net_delivered_qty"] = 0

    # ## Map delivered quantity as the same as quantity in the oms file
    # mask = df['final_status'] == 'delivered'
    # df.loc[mask, 'net_delivered_qty'] = df.loc[mask, 'quantity_oms']

    return df


def get_order_id_from_awb(cod_dict: dict, oms_df: pd.DataFrame) -> dict:
    """
    Gets order_id by mapping awb no from the oms file
    Arguments:
    1. cod_dict: dictionary of cod data
    2. oms_df: oms data
    """

    cod_dict = change_dict_keys_to_lower_case(cod_dict)

    oms_df["awbno"] = oms_df["awbno"].astype(str).str.lower().str.strip()
    awb_mapper = oms_df.groupby("awbno")["order_id"].first().to_dict()
    logistics_providers = list(cod_dict.keys())
    print(f"0.0 Total COD partners: {logistics_providers}")

    for provider in logistics_providers:
        print(f"0.1 {provider}")
        df = cod_dict.get(provider)
        if provider in ["bluedart"]:
            df.columns = df.columns.astype(str).str.strip().str.lower()
            df["awbno#"] = df["awbno#"].astype(str).str.lower().str.strip()
            mask = df["awbno#"].str.contains(".0")
            df.loc[mask, "awbno#"] = df.loc[mask, "awbno#"].str.split(".0").str[0]
            x = set(awb_mapper.keys())
            y = set(df["awbno#"])
            print(f"matching keys: {len(x & y)}")
            df["order_id"] = df["awbno#"].map(awb_mapper)
            cod_dict[provider] = df
        elif provider in ["delhivery"]:
            cols = list(df.columns.astype(str).str.lower().str.strip())
            if "order number" not in cols and "waybill_num" in cols:
                df["order number"] = df["waybill_num"].map(awb_mapper)
                cod_dict[provider] = df
        elif provider in ["dtdc"]:
            df["awbno"] = df["awbno"].astype(str).str.lower().str.strip()
            df["order_id"] = df["awbno"].map(awb_mapper)
            cod_dict[provider] = df

    return cod_dict


def standardize_cod(cod_dict: dict) -> pd.DataFrame:
    """
    Clean logistics file to get the right columns and rename them
    Arguments:
    1. cod_dict: a dictionary of all cod reconciliation files from all 3PL players
    """

    cod_dict = change_dict_keys_to_lower_case(cod_dict)

    rename_dict = {
        "shipdelight": [
            "orderno",
            "collected amount",
            "remitted amount",
            "remitted date",
        ],
        "delhivery": ["order number", "cod amount", "cod amount", "date"],
        "bluedart": ["order_id", "amount", "amount", "deposit date"],
        "dtdc": ["order_id", "amount", "remitted amount", "date"],
    }
    standard_col_names = [
        "order_id",
        "collected_amount",
        "remitted_amount",
        "remitted_date",
    ]

    new_3pl = [x for x in cod_dict if x not in rename_dict]
    list_of_dfs = []

    if new_3pl:
        warn_str = f"{new_3pl} not found in standardizations. Please add the customization in all codes"
        warnings.warn(warn_str)
        sys.exit(1)

    print(f"1.0 COD Partners: {cod_dict.keys()}")
    for logistics_name, temp_df in cod_dict.items():
        print(f"x. logistics_name: {logistics_name}")
        print(f"x. df for logistics_name {logistics_name}: {temp_df}")
        print(f"Data types for {logistics_name}: {temp_df.dtypes}")
        temp_df.columns = temp_df.columns.astype(str).str.lower().str.strip()
        temp_df = temp_df[rename_dict.get(logistics_name)]
        temp_df.columns = standard_col_names
        temp_df["3pl_cod_name"] = logistics_name
        mask = temp_df["order_id"].astype(str).str.contains("#")
        temp_df.loc[mask, "order_id"] = (
            temp_df.loc[mask, "order_id"].astype(str).str.split("#").str[1]
        )
        print(f"x.1. df for logistics_name {logistics_name}: {temp_df}")
        agg_func = {
            col: ("sum" if pd.api.types.is_numeric_dtype(temp_df[col]) else "first")
            for col in temp_df.columns
        }
        print(f"x.2. df for logistics_name {logistics_name}: {temp_df}")
        temp_df = temp_df.groupby("order_id", as_index=False).agg(agg_func)
        print(f"x.3. df for logistics_name {logistics_name}: {temp_df}")
        list_of_dfs.append(temp_df)

    df = pd.concat(list_of_dfs, axis=0)
    print(f"1. List of df's: {list_of_dfs}")
    print(f"2. df: {df}")
    df["order_id"] = df["order_id"].astype(str)

    return df


def clean_cod(cod_dict: dict, oms_df: pd.DataFrame) -> dict:
    """
    Make relevant changes to the cod file depending on the source.
    Arguments:
    1. cod_dict: A dictionary of cod files
    2. oms_df: OMS data read as a Dataframe
    """

    for name in cod_dict:
        df = cod_dict.get(name)
        df.columns = df.columns.astype(str).str.lower().str.strip()
        cols = list(df.columns)
        if name == "shipdelight":
            df["orderno"] = df["orderno"].where(
                ~df["orderno"].astype(str).str.contains("#"),
                df["orderno"].astype(str).str.strip("#").str[1],
            )
        elif name == "delhivery":
            mask = df["order number"].astype(str).str.contains("_")
            df.loc[mask, "order number"] = (
                df.loc[mask, "order number"].astype(str).str.split("_").str[0]
            )
            if "date" in cols:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            else:
                df["date"] = pd.NaT
        elif name == "dtdc":
            df["date"] = pd.NaT

        cod_dict[name] = df

    return cod_dict


# def add_order_number_to_delhivery(df: pd.DataFrame, oms_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Sometimes, delhivery provides a file that does not contain the required columns. We need to modify the columns to add the relevant fields
#     Arguments:
#     1. df: Delhivery COD dataframe
#     2. oms_df: OMS data as a dataframe
#     """
#     order_id_mapper = oms_df.groupby('order_id', as_index=False)['awbno'].first().to_dict()
#     df['order number'] = df['waybill_num'].map(order_id_mapper)

#     return df


def allocate_cod_collection_to_sub_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function allocates cod collections and remittances to sub-order
    Arguments:
    a. df: standardized and mapped order_export file from shopify
    """

    df["collected_amount"] = df["collected_amount"].astype(float)
    df["remitted_amount"] = df["remitted_amount"].astype(float)
    df["order_pct_allocation"] = df["order_pct_allocation"].astype(float)

    df = df.assign(
        cod_collections_allocated=lambda x: x["collected_amount"]
        * x["order_pct_allocation"],
        cod_remittances_allocated=lambda x: x["remitted_amount"]
        * x["order_pct_allocation"],
    )

    return df


def create_amount_collected_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new dataframe with total amount collected against each order id. We will use this table to identify
            (a) prepaid/cod/part-paid orders
            (b) to identify settled/unsettled orders

    To this dataframe we will add payment gateway collections and remittances, to arrive at the final mismatches
    NOTE: This has to be done at an order level and not at a sub-order level - becauase the 3pl partner only has order level details

    Arguments:
    df: A dataframe of order file mapped with the cod collection file
    """

    ## Create a grouped dataframe containing only the total gross sale, cod collected amount and cod remitted amount.
    req_cols = [
        "cod_collections_allocated",
        "cod_remittances_allocated",
        "pg_collections_allocated",
        "pg_remittances_allocated",
        "gross_sales",
        "disc_allocated",
        "shipping_allocated",
        "amount_to_be_collected",
        "refund_allocated",
        "remitted_date",
    ]
    agg_func = {col: ("first" if col == "remitted_date" else "sum") for col in req_cols}
    remitted_df = df.groupby("order_id", as_index=False).agg(agg_func).fillna(0)
    rename_dict = {
        "cod_collections_allocated": "cod_collection",
        "cod_remittances_allocated": "cod_remittances",
        "pg_collections_allocated": "pg_collection",
        "pg_remittances_allocated": "pg_remittances",
        "refund_allocated": "amt_refunded",
    }
    remitted_df.rename(columns=rename_dict, inplace=True)

    return remitted_df


def add_order_status_to_remitted(
    df: pd.DataFrame, order_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function adds final order status to each order
    Arguments:
    1. df: remittances_df. A dataframe summarizing order status (settled/unsettled) for all order ID's
    2. order_df: mapped orders files that contains a minimum of order_id and order_status
    """

    status_mapper = order_df.groupby("order_id")["final_status"].first().to_dict()
    df["final_status"] = df["order_id"].map(status_mapper)

    return df


def add_pg_name_to_remitted(df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds payment gateway name to the remitted dataframe
    Arguments:
    1. df: remitted dataframe
    2. order_df: dataframe of mapped_orders, containing the name of the payment gateway
    """

    pg_mapper = order_df.groupby("order_id")["payment_gateway"].first().to_dict()
    df["payment_gateway"] = df["order_id"].map(pg_mapper)

    return df


def add_3pl_name_to_remitted(df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds 3pl name to the remitted dataframe
    Arguments:
    1. df: remitted dataframe
    2. order_df: dataframe of mapped_orders, containing the name of the 3pl provider
    """

    logistics_mapper = order_df.groupby("order_id")["3pl_cod_name"].first().to_dict()
    df["3pl_cod_name"] = df["order_id"].map(logistics_mapper)

    return df


def add_payment_method_to_remitted(
    df: pd.DataFrame, order_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function adds 3pl name to the remitted dataframe
    Arguments:
    1. df: remitted dataframe
    2. order_df: dataframe of mapped_orders, containing the name of the 3pl provider
    """

    payment_method_mapper = (
        order_df.groupby("order_id")["payment_method"].first().to_dict()
    )
    df["payment_method"] = df["order_id"].map(payment_method_mapper)

    return df


def standardize_pg(pg_dict: dict) -> pd.DataFrame:
    """
    This function accepts all payment gaateway files as a dictionary of dataframes of the form: {pg1: pg_df1, pg2: pg_df2.....}. reads individual dataframe.
    Then using a dictionary of standard column names, extracts each pg_df and standardizes it based on the dataframe
    Arguments:
    a.pg_dict: A dictionary of all payment gateway dataframes. Just read the excel with multiple sheets as a dataframe
    """

    pg_dict = change_dict_keys_to_lower_case(pg_dict)

    pg_col_selector = {
        "razorpay": [
            "transaction_entity",
            "amount",
            "fee (exclusive tax)",
            "tax",
            "debit",
            "credit",
            "payment_method",
            "settled_at",
            "settled_by",
            "order_receipt",
            "currency",
            "order_notes",
            "order_id",
        ],
        "gokwik": [
            "transaction type",
            "amount",
            "fee",
            "tax",
            "debit",
            "credit",
            "payment method",
            "settlement date",
            "settled by",
            "transaction rrn",
            "currency",
            "shopify order id",
            "order_id",
        ],
        "payu": [
            "status",
            "merchant requested fee",
            "total processing fees",
            "total service tax",
            "credit",
            "debit",
            "payment type",
            "settlement date",
            "payment source",
            "merchant txn id",
            "settlement currency",
            "shopify_order_id",
            "order_id",
        ],
    }

    pg_standardizer = [
        "type",
        "amount",
        "pg_charges",
        "tax",
        "debit",
        "credit",
        "payment_method",
        "settled_at",
        "settled_by",
        "pg_reference",
        "currency",
        "order_notes",
        "order_id",
    ]

    new_pgs = [x for x in pg_dict.keys() if x not in pg_col_selector]
    list_of_dfs = []
    if new_pgs:
        warn_str = f"{new_pgs} not found in standardizer, write the logic for new payment gateways"
        warnings.warn(warn_str)
        sys.exit(1)

    for pg in pg_dict.keys():
        df = pg_dict.get(pg)
        df.columns = df.columns.astype(str).str.lower().str.strip()
        if pg == "gokwik":
            df["fee"] = df["fee"] + df["additional fees"]
            df["tax"] = df["tax"] + df["additional tax"]
            ## We will have to check which of the two work
            # df['order_id'] = df['shopify order id']
            df["order_id"] = df["shopify order id"].astype(str).str.split("#").str[1]
            pg_dict[pg] = df

        elif pg == "razorpay":
            pattern = r'shopify_order_id":"#(.*?)"'
            df["order_id"] = df["order_notes"].str.extract(pattern)

        elif pg == "payu":
            df["credit"] = pd.NA
            df["debit"] = pd.NA
            df["shopify_order_id"] = pd.NA
            df["order_id"] = pd.NA

        req_cols = pg_col_selector.get(pg)
        renamed_cols = pg_standardizer
        df = df[req_cols]
        df.columns = renamed_cols
        df["payment_gateway"] = pg
        mask = df["type"] == "refund"
        df.loc[mask, "amt_refunded"] = df.loc[mask, "amount"]
        df.loc[mask, "amount"] = 0
        df["amt_refunded"] = df["amt_refunded"].fillna(0)
        df["type"] = df["type"].astype(str).str.lower().str.strip()
        num_cols = ["amount", "pg_charges", "tax", "debit", "credit", "amt_refunded"]
        agg_func = {k: ("sum" if k in num_cols else "first") for k in df.columns}
        df = df.groupby("order_id", as_index=False).agg(agg_func)
        list_of_dfs.append(df)

    pg_df = pd.concat(list_of_dfs, axis=0)

    return pg_df


def clean_pg(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans payment gateway files, by correcting for refunds
    Arguments:
    a. Standardized payment gateway files
    """

    pgs = list(set(df["payment_gateway"]))
    list_of_dfs = []

    for pg in pgs:
        mask = df["payment_gateway"] == pg
        temp_df = df[mask].copy()
        str_cols = [
            "type",
            "payment_method",
            "settled_at",
            "settled_by",
            "pg_reference",
            "currency",
            "payment_gateway",
            "pg_reference",
        ]
        agg_func = {x: ("first" if x in str_cols else "sum") for x in temp_df.columns}
        temp_df = temp_df.groupby("order_id", as_index=False).agg(agg_func)
        temp_df["settled_at"] = pd.to_datetime(
            temp_df["settled_at"], errors="coerce", dayfirst=True
        )
        list_of_dfs.append(temp_df)

    if list_of_dfs:
        df = pd.concat(list_of_dfs, axis=0)

    rename_dict = {
        "amount": "gross_pg_amount_collected",
        "amt_refunded": "total_pg_refunded",
        "tax": "total_tax_on_pg_charges",
        "settled_at": "pg_settled_at",
        "settled_by": "pg_settled_by",
        "debit": "total_pg_debits",
        "credit": "total_pg_credits",
        "currency": "pg_currency",
    }
    df.rename(columns=rename_dict, inplace=True)
    df["total_pg_amount_collected"] = (
        df["gross_pg_amount_collected"] - df["total_pg_refunded"]
    )
    df["total_pg_amount_remitted"] = df["total_pg_amount_collected"]

    return df


def extract_order_id_for_payu(df: pd.DataFrame, pg_df: pd.DataFrame) -> pd.DataFrame:
    """
    PayU has merchant transaction ID. We need to map this with the shopify order ID to get the order_id field
    Arguments:
    a. df: standardized and mapped order_export file from shopify
    b. pg_df: standardized PG file
    """

    oid_dict = df.groupby("pg_reference")["order_id"].first().to_dict()
    mask = pg_df["payment_gateway"] == "payu"
    pg_df.loc[mask, "order_id"] = pg_df.loc[mask, "pg_reference"].map(oid_dict)
    pg_df.loc[mask, "shopify_order_id"] = pg_df.loc[mask, "pg_reference"].map(oid_dict)

    return pg_df


def allocate_pg_to_suborder(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function allocates pg charges against each order_id
    Arguments:
    a. df: standardized and mapped order_export file from shopify
    """

    df = df.assign(
        pg_collections_allocated=lambda x: x["total_pg_amount_collected"]
        * x["order_pct_allocation"],
        pg_remittances_allocated=lambda x: x["total_pg_amount_remitted"]
        * x["order_pct_allocation"],
        pg_charges_allocated=lambda x: x["pg_charges"] * x["order_pct_allocation"],
        pg_taxes_allocated=lambda x: x["total_tax_on_pg_charges"]
        * x["order_pct_allocation"],
    )

    return df


def calculate_settled_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the total amount collected and remitted against each order and subsequently assigns a "settled / unsettled" status against each order.
    There are multiple statuses based on criteria.
    1. "settled": total_remittances = amount_to_be_collected and total_remittances != 0
    2. "amount_mismatch_remittance": total_collections = amount_to_be_collected and total_remittances != amount_to_be_collected and total_remittances != 0
    3. "unsettled_collected_not_remitted": total_collections = amount_to_be_collected and total_remittances = 0
    4. "amount_mismatch_collection": total_collections != amount_to_be_collected and total_collections != 0
    5. "unsettled": total_collections = 0
    6. if amt_refunded > 0 and total_collections > 0 and amt_refunded + total_collections

    Arguments:
    a. df: remittance dataframe
    """

    # df['amount_to_be_collected'] = df['amount_to_be_collected'].replace("no_matches", 0)
    # df['amount_to_be_collected'] = df['amount_to_be_collected'].replace("no_matchesno_matches", 0)
    # df['cod_collection'] = df['cod_collection'].replace("no_matches", 0)
    # df['pg_collection'] = df['pg_collection'].replace("no_matches", 0)
    # df['cod_remittances'] = df['cod_remittances'].replace("no_matches", 0)
    # df['pg_remittances'] = df['pg_remittances'].replace("no_matches", 0)

    df["amount_to_be_collected"] = df["amount_to_be_collected"].astype(float)
    df["cod_collection"] = df["cod_collection"].astype(float)
    df["cod_remittances"] = df["cod_remittances"].astype(float)
    df["pg_collection"] = df["pg_collection"].astype(float)
    df["pg_remittances"] = df["pg_remittances"].astype(float)
    df = df.assign(
        total_collections=lambda x: x["cod_collection"] + x["pg_collection"],
        total_remittances=lambda x: x["cod_remittances"] + x["pg_remittances"],
        pending_collections=lambda x: x["amount_to_be_collected"]
        - x["total_collections"],
        pending_remittances=lambda x: x["total_collections"] - x["total_remittances"],
    )

    ## Round off the column so remove erroneous statuses
    cols_to_round = [
        "total_collections",
        "total_remittances",
        "pending_collections",
        "pending_remittances",
        "amount_to_be_collected",
        "amt_refunded",
    ]
    df[cols_to_round] = df[cols_to_round].round(2)

    ## Condition 1: settled: total_remittances = amount_to_be_collected and total_remittances != 0
    mask = df["total_remittances"] == df["amount_to_be_collected"]
    df.loc[mask, "payment_status"] = "settled"

    ## Condition 2: "amount_mismatch_remittance": total_collections = amount_to_be_collected and total_remittances != amount_to_be_collected and total_remittances != 0
    mask = (
        (df["total_collections"] == df["amount_to_be_collected"])
        & (df["total_remittances"] != df["amount_to_be_collected"])
        & (df["total_remittances"] != 0)
    )
    df.loc[mask, "payment_status"] = "amount_mismatch_remittance"

    ## Condition 3: "unsettled_collected_not_remitted": total_collections = amount_to_be_collected and total_remittances = 0
    mask = (df["total_collections"] == df["amount_to_be_collected"]) & (
        df["total_remittances"] == 0
    )
    df.loc[mask, "payment_status"] = "unsettled_collected_not_remitted"

    ## Condition 4: "amount_mismatch_collection": total_collections != amount_to_be_collected and total_collections != 0
    mask = (df["total_collections"] != df["amount_to_be_collected"]) & (
        df["total_collections"] != 0
    )
    df.loc[mask, "payment_status"] = "amount_mismatch_collection"

    ## Condition 5: "unsettled": total_collections = 0
    mask = df["total_collections"] == 0
    df.loc[mask, "payment_status"] = "unsettled"

    return df


def allocate_logistics_cost(
    df: pd.DataFrame, logistics_standard_col_names: list
) -> pd.DataFrame:
    """
    This function maps logistics costs to each sub-order
    Arguments:
    a. df: mapped order_export file from shopify order_export
    b. logistics_standard_col_name: List of standard column names based on all logistics files
    """

    allocate_cols = [
        "net_logistics_cost",
        "cgst",
        "sgst",
        "igst",
        "gross_total_logistics",
    ]
    missing_logistics_cols = [
        x for x in allocate_cols if x not in logistics_standard_col_names
    ]

    if missing_logistics_cols:
        warn_str = f"{missing_logistics_cols} not found in logistics file. Please edit logistics_standard_col_names variable in standardize_logistics function to reflect the correct columns."
        warnings.warn(warn_str)

    df.loc[:, allocate_cols] = df[allocate_cols].mul(df["order_pct_allocation"], axis=0)

    return df


def add_daily_marketing_spends(
    df: pd.DataFrame,
    order_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Allocate Daily marketing spend to sub-orders
    Arguments:
    a. df: daily_marketing_spends dataframe. Should contain the following columns (i) date (ii) "marketing_spends"
    b. order_df: shopify order_export file
    """

    mask = order_df["order_date"] == "no_matches"
    order_df.loc[mask, "date"] = order_df.loc[mask, "order_date"].dt.date
    daily_orders = order_df.groupby("date")["quantity_ordered"].sum()

    if df is None or df.empty:
        order_df["marketing_spends"] = 0
    else:
        df["mkt_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.groupby("date", as_index=False)["marketing_spends"].sum()
        order_df = order_df.merge(
            df,
            left_on="date",
            right_on="mkt_date",
            how="left",
            suffixes=["", "_marketing"],
        )

    return order_df


def prepare_PnL(df: pd.DataFrame, incl_tax: bool, mis_bool: bool) -> pd.DataFrame:
    """
    Calculate P&L - by calculating CM1, CM2 and CM3
    """

    pg_cols = ["pg_charges_allocated", "pg_taxes_allocated"]
    cogs_cols = ["cogs"]
    logistics_cols = ["net_logistics_cost", "cgst", "sgst", "igst"]
    marketing_cols = ["marketing_spends"]

    if mis_bool:
        pg_cols = ["est_" + x for x in pg_cols]
        cogs_cols = ["est_" + x for x in cogs_cols]
        logistics_cols = ["est_" + x for x in logistics_cols]

    warn_str = (
        f"If calculating without taxes, ensure COGS and marketing are before tax amount"
    )
    warnings.warn(warn_str)

    if incl_tax:
        df["topline_sales"] = df["gross_sales"]
        df["cm1_cogs"] = df["cogs"]
        df["cm2_logistics"] = df[logistics_cols].sum(axis=1)
        df["cm2_pg"] = df[pg_cols].sum(axis=1)
        df["cm3_mkt"] = df[marketing_cols].sum(axis=1)
    else:
        df["topline_sales"] = df["net_sales"]
        df["cm1_cogs"] = df["cogs"]
        df["cm2_logistics"] = df["net_logistics_cost"]
        df["cm2_pg"] = df["pg_charges_allocated"]
        df["cm3_mkt"] = df[marketing_cols].sum(axis=1)

    df["cm1"] = df["topline_sales"] - df["cm1_cogs"]
    df["cm2"] = df["cm1"] - df["cm2_pg"] - df["cm2_logistics"]
    df["cm3"] = df["cm2"] - df["cm3_mkt"]

    return df


def estimate_missing_cogs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate and add missing cogs. Missing cogs to be estimated as an average of average value of cogs
    Arguments:
    a. df: mapped shopify order_export dataframe
    """

    est_cogs = df.groupby("sku")["cogs_per_unit"].mean()
    average_cogs = est_cogs.mean()
    est_cogs_dict = est_cogs.fillna(average_cogs).to_dict()

    mask = df["cogs_per_unit"].isna()
    df.loc[mask, "cogs_per_unit"] = df.loc[mask, "sku"].map(est_cogs_dict)
    df["cog_est"] = df["cogs_per_unit"] * df["net_delivered_qty"]

    return df


def estimate_missing_pg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate and add missing payment gateway charges. Unsettled pg charges will be taken as a %age of the gross sale values
    Arguments:
    a. df: mapped shopify order_export dataframe
    """

    pg_pct = df["pg_charges_allocated"] / df["gross_sales"]
    average_pg = df["pg_pct"].mean()
    mask = (df["pg_charges_allocated"].isna()) | (df["pg_charges_allocated"] == 0)
    df.loc[mask, "est_pg_charges_allocated"] = df.loc[mask, "gross_sales"] * average_pg
    df.loc[mask, "pg_taxes_allocated"] = df.loc[mask, "est_pg_charges_allocated"] * 0.18

    return df


def estimate_missing_logistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate and add missing logistics charges. Logistics charges are estimated by taking the average cost per unit
    Arguments:
    a. df: mapped shopify order_export dataframe
    """

    est_net_logistics_per_unit = df["net_logistics_cost"] / df["quantity_ordered"]
    est_gross_logistics_per_unit = est_net_logistics_per_unit * 1.18
    mask = (df["net_logistics_cost"].isna()) | (df["net_logistics_cost"] == 0)
    df.loc[mask, "est_net_logistics_cost"] = (
        df.loc[mask, "quantity_ordered"] * est_net_logistics_per_unit
    )
    df.loc[mask, "est_igst"] = df.loc[mask, "est_net_logistics_cost"] * 0.18
    df.loc[mask, "est_cgst"] = 0
    df.loc[mask, "est_sgst"] = 0

    return df


def map_shopify_payments(
    orders: pd.DataFrame,
    oms_df: pd.DataFrame,
    cod_dict: dict,
    pg_dict: dict,
    logistics_dict: dict,
    cogs_df: pd.DataFrame,
    daily_marketing_df: Optional[pd.DataFrame] = None,
    shopify_payments_df: Optional[pd.DataFrame] = None,
    currency_df: Optional[pd.DataFrame] = None,
    incl_tax: bool = True,
    mis_bool: bool = False,
) -> pd.DataFrame:
    """
    This function reconciles shopify payments and associated costs to the orders. By reconciliation we mean whether all associated amounts (inflows) have been received against an order.
    Arguments:
    1. order_df: Shopify order file. Refer section on Shopify code and use that same code to generate the order file (change dates)
    2. oms_df: sale file from order management system . Currently only supports Unicommerce and EasyEcom (Use "Sale Orders" file for "Unicommerce")
    3. cod_dict : COD settlement files from relevent 3pl logistics providers read as a dictionary. Keys: 3pl name, value dataframe of order Id wise cod collection and remittances.
    4. pg_dict: PG settlement files read as a dictionary. keys: payment gateway name, values: dataframe of datewise settlements
    5. logistics_dict: orderid wise total logistics cost read as a dictionary. Keys: 3pl name, value dataframe of order Id wise logistics charges
    6. cogs_df: File mapping cost of goods sold. Taken directly from Additional Data sheet of Reconzo with same column names
    7. daily_marketing_df: a Dataframe containing daily marketing spends. Can support a maximum of 5 columns - date, meta, google, others and total.
    8. shopify_payments_df (optional): shopify payments (usually valid for US sales - not sure so haven't added that yet). This is a standard shopify file (Finance>Payouts). Default = None
    9. currency_df (optional): datewise currency converter (if USD to INR is required, get daily exchange rates from https://in.investing.com/currencies/usd-inr-historical-data)
    10. incl_tax (Optional): Boolean variable to calculate with or without tax. Default value is True
    11. mis_bool (Optional): Boolean variable to calculate as MIS or as payment recon

    Note: pg_df will be required when shopify_payments_df is None. Essentially, there has to be atleast one place from which payments are gathered and reconciled.

    Returns a dataframe showing the reconciliation and associated payments prepared for the MIS

    Output:
    -------
    """
    warnings.warn(
        f"Always ensure orderID's in both OMS and Shopify files are converted to the same type. \nThis code does not process order ID. If match is not found, the code witll throw an error\n"
    )
    print(f"{'*' * 10}")

    logistics_providers = ["shipdelight", "delhivery", "bluedart", "dtdc"]
    logistics_name = list(logistics_dict.keys())
    missing_logistics = [x for x in logistics_name if x not in logistics_providers]
    if missing_logistics:
        warnings.warn(
            f"Logistics provider names not in standard list for {missing_logistics}. Output can be unexpected. pre-process all logistics file as per documentation"
        )

    if daily_marketing_df is not None and not daily_marketing_df.empty:
        if "marketing_spends" not in daily_marketing_df.columns:
            warn_str = f'"marketing_spends" not found in daily_marketing_df columns'
    # else:
    #     some_dict = {"date": ["01-01-2025"], "marketing_spends": [0]}
    #     daily_marketing_df = pd.DataFrame(some_dict)

    orders.columns = orders.columns.astype(str).str.lower().str.strip()
    oms_df.columns = oms_df.columns.astype(str).str.strip().str.lower()
    # cod_df.columns = cod_df.columns.astype(str).str.strip().str.lower()

    ## Preprocess and standardize order_df and allocate shipping charges
    print(f"{'*' * 10}Standardizing orders and calculating amount to be collected...")
    order_df = standardize_orders(orders)
    order_df = add_payment_reference_number_for_each_order(order_df)
    order_df = calculate_order_pct_allocation(order_df)
    order_df = allocate_discounts_to_suborder(order_df)
    order_df = allocate_shipping_to_suborder(order_df)
    order_df = allocate_refunds_to_suborder(order_df)
    order_df = calculate_amount_to_be_collected(order_df)
    print(f"{'*' * 10}Orders standardized.")

    ## Preprocess and standardize oms
    print(
        f"{'*' * 10}Standardizing OMS and establishing final order status and net delivered quantity..."
    )
    oms_name = identify_oms(oms_df)
    oms_rename_dict = get_required_oms_columns(oms_name)
    missing_oms_cols = [c for c in oms_rename_dict if c not in oms_df.columns]
    if missing_oms_cols:
        warnings.warn(
            f"{missing_oms_cols} not found in oms_df. Recheck oms_data and rerun the code",
            UserWarning,
        )
    oms_df = clean_oms(oms_df, oms_name)
    oms_df = standardize_oms(oms_df, oms_name, oms_rename_dict)
    oms_df = assign_shipping_status(oms_df)
    oms_df = assign_order_status(oms_df)
    oms_df = assign_final_status_as_list(oms_df)
    oms_df = aggregate_oms(oms_df)
    oms_df = assign_final_status(oms_df)

    ## Merge orders with oms to get final status and net delivered quantity
    oms_keys = set(oms_df["key"])
    orders_keys = set(order_df["key"])
    order_df = order_df.merge(oms_df, on="key", how="left", suffixes=["", "_oms"])
    if order_df.index.size != len(orders_keys):
        warn_str = f"Duplicates found in merge: len of order_df{len(orders_keys)} len of oms_df: {len(oms_keys)}"
        warnings.warn(warn_str)
        sys.exit(1)

    order_df["order_date"] = pd.to_datetime(
        order_df["order_date"].replace("no_matches", pd.NaT), errors="coerce"
    )
    order_df["order_pct_allocation"] = order_df["order_pct_allocation"].replace(
        "no_matches", 1
    )
    order_df = calculate_net_delivered_qty(order_df)
    print(f"{'*' * 10}\nNet delivered quantity calculated")

    ## Get logistics file (COD remittances), rename columns, clean the file and map it to orders_df
    print(f"{'*' * 10}\nEstablishing Cash on Delivery amount collected and remitted...")
    cod_dict = get_order_id_from_awb(cod_dict, oms_df)
    cod_dict = clean_cod(cod_dict, oms_df)
    cod_df = standardize_cod(cod_dict)
    order_df = order_df.merge(
        cod_df, on="order_id", how="left", suffixes=["", "_cod"]
    ).fillna(0)
    order_df = allocate_cod_collection_to_sub_orders(order_df)

    ## pre-process and standardize pg
    print(f"{'*' * 10}\nEstablishing Payment Gateway amount collected and remitted...")
    pg_df = standardize_pg(pg_dict)
    pg_df = clean_pg(pg_df)
    pg_df = extract_order_id_for_payu(order_df, pg_df)

    order_df = order_df.merge(pg_df, on="order_id", how="left", suffixes=["", "_pg"])
    order_df = allocate_pg_to_suborder(order_df)
    print(f"{'*' * 10}Payment collected and remitted mapped.")

    ## Idenitfy unsettled orders
    remitted_df = create_amount_collected_df(order_df)
    remitted_df = calculate_settled_orders(remitted_df)
    remitted_df = add_order_status_to_remitted(remitted_df, order_df)
    remitted_df = add_pg_name_to_remitted(remitted_df, order_df)
    remitted_df = add_3pl_name_to_remitted(remitted_df, order_df)
    remitted_df = add_payment_method_to_remitted(remitted_df, order_df)
    settled_status_dict = remitted_df.set_index("order_id")["payment_status"].to_dict()
    order_df["payment_status"] = order_df["order_id"].map(settled_status_dict)

    print(f"{'*' * 10}Order settlement status identified.")

    ## Get logistics costs
    print(f"{'*' * 10}Mapping logistics cost to each suborder...")
    logistics_dict, awb_mapper = clean_logistics(
        logistics_dict, logistics_providers, oms_df
    )
    logistics_df, logistics_standard_col_names = standardize_logistics(logistics_dict)
    order_df = order_df.merge(
        logistics_df, on="order_id", how="left", suffixes=["", "_logistics"]
    )
    order_df = allocate_logistics_cost(order_df, logistics_standard_col_names)

    ## Map COGS
    print(f"{'*' * 10}Mapping cogs to each suborder...")
    cogs_df = clean_cogs(cogs_df)
    cogs_df["cogs_per_unit"] = cogs_df["cogs_per_unit"].fillna(0)
    cogs_df["sku"] = cogs_df["sku"].astype(str)
    order_df["sku"] = order_df["sku"].astype(str)

    order_df["month"] = order_df["month"].astype(str)
    cogs_df["month"] = cogs_df["month"].astype(str)

    order_df = order_df.merge(
        cogs_df, on=["sku", "month"], how="left", suffixes=["", "_cogs"]
    )
    order_df["cogs"] = (
        order_df["cogs_per_unit"].fillna(0) * order_df["quantity_ordered"]
    )

    ## Calculate P&L
    print(f"{'*' * 10}Calculating P&L...")
    order_df["order_date"] = pd.to_datetime(order_df["order_date"], errors="coerce")
    order_df = add_daily_marketing_spends(daily_marketing_df, order_df)

    if mis_bool:
        order_df = estimate_missing_cogs(order_df)
        order_df = estimate_missing_pg(order_df)
        order_df = estimate_missing_logistics(order_df)

    order_df = prepare_PnL(order_df, incl_tax, mis_bool)
    print(f"{'*' * 10}Code executed successfully!!")

    return (
        order_df,
        oms_df,
        cod_df,
        remitted_df,
        pg_df,
        logistics_df,
        cogs_df,
        awb_mapper,
    )


### ----- USE SHOPIFY order_export FILE FOR THIS PIECE OF CODE
