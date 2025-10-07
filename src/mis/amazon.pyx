import pandas as pd
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from pandas.tseries.offsets import DateOffset


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
    elif file.name.endswith("zip"):
        try:
            df = pd.read_csv(file, compression="zip")
        except KeyError:
            df = pd.read_excel(file, engine="openpyxl")

    else:
        msg = f"File Extension {file.name.split('.')[1]} not found."
        warnings.warn(msg)
        return None

    return df


def ingest_files(_path_: str) -> dict:
    """
    Loops through a folder to read the relevant files and create an MIS
    """

    data_dict = defaultdict(pd.DataFrame)

    name_formats = {
        "agg_payments": "payments",
        "agg_returns": "returns",
        "agg_replacements": "replacements",
        "agg_returns-mfn": "returns-mfn",
        "b2bReport": "mtr",
        "b2cReport": "mtr",
    }

    folder_path = Path(_path_)
    for file in folder_path.iterdir():
        if ".DS_Store" in file.name:
            continue
        else:
            print(f"Processing file: {file}")
            df = read_any_file_type(file)
            df.columns = df.columns.astype(str).str.lower().str.strip()
            if any(x in file.stem for x in ["b2bReport", "b2cReport"]):
                df["report_type"] = file.name[:3]
                month = file.name.split("_")[1]
                year = file.name.split("_")[2].split(".")[0]
                month_year = f"01-{month}-{year}"
                df["month_year"] = datetime.strptime(month_year, "%d-%B-%Y")
                df["month_year"] = pd.to_datetime(
                    df["month_year"], errors="coerce", dayfirst=True
                ).dt.date

            file_types = [x for x in name_formats if x in file.stem]
            if file_types:
                file_type = max(file_types, key=len)
                if name_formats.get(file_type) in data_dict:
                    temp_df = data_dict.get(name_formats.get(file_type))
                    df = pd.concat([temp_df, df], axis=0)

                data_dict[name_formats.get(file_type)] = df

    data_dict = dict(data_dict)
    return data_dict


def tag_current_month_orders(data_dict: dict) -> dict:
    """
    Tags each MTR order with current or previous month tag
    """

    mtr = data_dict["mtr"]
    date_cols = ["order date", "invoice date", "month_year"]
    for col in date_cols:
        mtr[col] = pd.to_datetime(mtr[col], errors="coerce")

    mask = mtr["order date"].isna()
    mtr.loc[mask, "order date"] = mtr.loc[mask, "invoice date"]

    mtr["month_year_end"] = mtr["month_year"] + DateOffset(months=1)
    mask = (mtr["order date"] >= mtr["month_year"]) & (
        mtr["order date"] < mtr["month_year_end"]
    )
    mtr.loc[mask, "current_month_tag"] = "current_month"
    mtr.loc[~mask, "current_month_tag"] = "previous_month"

    data_dict["mtr"] = mtr

    return data_dict


def add_gross_sales(data_dict: dict) -> dict:
    """
    Adds a Gross sale number to each sale
    """

    mtr = data_dict["mtr"]
    asins = set(mtr["asin"])
    gross_sale_dict = {
        k: mtr[mtr["asin"] == k]["principal amount"].mode().iloc[0] for k in asins
    }
    mtr["gross_sale_mis"] = mtr["asin"].map(gross_sale_dict)
    data_dict["mtr"] = mtr

    return data_dict


def add_gross_quantity(data_dict: dict) -> dict:
    """
    Adds a Gross sale number to each sale
    """

    mtr = data_dict["mtr"]
    mask = mtr["quantity"] < 1
    mtr.loc[mask, "gross_quantity_mis"] = 1
    mtr.loc[~mask, "gross_quantity_mis"] = mtr.loc[~mask, "quantity"]
    data_dict["mtr"] = mtr

    return data_dict


def create_monthly_payments(data_dict: dict) -> dict:
    """
    Breaks up payment file into monthly payments
    """

    payments = data_dict["payments"]
    mtr = data_dict["mtr"]

    payments["posted-date-time"] = pd.to_datetime(
        payments["posted-date-time"], errors="coerce", dayfirst=True
    ).dt.tz_convert("Asia/Kolkata")
    payments["posted-date-time"] = payments["posted-date-time"].dt.tz_localize(None)
    month_years = set(mtr["month_year"])

    for m in month_years:
        end_month = m + pd.DateOffset(months=1)
        mask = (payments["posted-date-time"] >= m) & (
            payments["posted-date-time"] < end_month
        )
        payments.loc[mask, "month_year"] = m
        payments.loc[mask, "month_year_end"] = end_month

    payments["month_year"] = pd.to_datetime(
        payments["month_year"], errors="coerce", dayfirst=True
    ).dt.date
    data_dict["payments"] = payments

    return data_dict


def map_sku_to_asin(data_dict: dict) -> dict:
    """ """
    mtr = data_dict["mtr"].copy()
    payments = data_dict["payments"].copy()

    asin_mapper = mtr.groupby("sku")["asin"].first().to_dict()
    payments["asin"] = payments["sku"].map(asin_mapper)

    data_dict["mtr"] = mtr
    data_dict["payments"] = payments

    return data_dict


def create_keys(data_dict: dict) -> dict:
    """ """

    mtr = data_dict["mtr"].copy()
    payments = data_dict["payments"].copy()

    mtr["key"] = mtr["order id"].astype(str) + mtr["asin"].astype(str)
    payments["key"] = payments["order-id"].astype(str) + payments["asin"].astype(str)

    data_dict["mtr"] = mtr
    data_dict["payments"] = payments

    return data_dict


def add_standard_statuses(data_dict: dict) -> dict:
    """ """
    mtr = data_dict["mtr"]
    mtr["transaction type"].replace("Shipment", "Net Delivered", inplace=True)
    mtr["transaction type"].replace("Cancel", "Cancelled", inplace=True)
    mtr["transaction type"].replace("Refund", "Return + RTO", inplace=True)
    mtr["transaction type"].replace("FreeReplacement", "Replacements", inplace=True)

    data_dict["mtr"] = mtr

    return data_dict


def create_mtr_pivots(
    data_dict: dict, col_to_pivot: str, order_status_mask: str
) -> dict:
    """
    Counts the total orders sold in the month
    """

    mtr = data_dict["mtr"]
    if order_status_mask == "Gross Sale":
        mask = mtr["transaction type"] != "Replacements"
        temp_df = mtr[mask].copy()
    else:
        mask = mtr["transaction type"] == order_status_mask
        temp_df = mtr[mask].copy()

    df = pd.pivot_table(
        temp_df, columns="month_year", index="key", values=col_to_pivot, aggfunc="sum"
    ).fillna(0)
    df = df.reset_index()
    df_name = col_to_pivot + order_status_mask
    data_dict[df_name] = df

    return data_dict


def create_mis_structure(data_dict: dict) -> dict:
    """ """

    ## Calculate units
    statuses = ["Gross Sale", "Return + RTO", "Cancelled", "Net Delivered"]
    df = pd.DataFrame()

    for status in statuses:
        df_name = "gross_quantity_mis" + status
        temp_df = data_dict[df_name].copy()
        some_dict = {
            col: [temp_df[col].sum()] for col in temp_df.columns if col != "key"
        }
        temp_df = pd.DataFrame(some_dict, index=[status])
        df = pd.concat([temp_df, df], axis=0)

        df_name = "gross_sale_mis" + status
        temp_df = data_dict[df_name].copy()
        some_dict = {
            col: [temp_df[col].sum()] for col in temp_df.columns if col != "key"
        }
        temp_df = pd.DataFrame(some_dict, index=[status + " - sales"])
        df = pd.concat([temp_df, df], axis=0)

    data_dict["mis"] = df
    return data_dict


def break_payments_by_month(data_dict: dict) -> dict:
    """
    Breaks payment files by month, so that overheads can be calculated separately for each month.
    """

    payments = data_dict["payments"]
    mtr = data_dict["mtr"]
    mtr["month_year"] = mtr["month_year"].dt.date

    months = set(mtr["month_year"])
    payments_monthly_dict = {}
    for m in months:
        mask = payments["month_year"] == m
        payments_monthly_dict[m] = payments[mask].copy()

    data_dict["payments_split"] = payments_monthly_dict
    data_dict["mtr"] = mtr

    return data_dict


def tag_sku_and_orders(data_dict: dict) -> dict:
    """ """
    payments = data_dict["payments"]
    order_pattern1 = r"^\d{3}-\d{7}-\d{7}$"
    order_pattern2 = r"^s\d{2}-\d{7}-\d{7}$"

    mask = payments["order-id"].astype(str).str.contains(
        order_pattern1, regex=True
    ) | payments["order-id"].astype(str).str.contains(order_pattern2, regex=True)
    payments.loc[mask, "valid_orderid"] = True
    payments.loc[~mask, "valid_orderid"] = False

    mask = payments["sku"].isna()
    payments.loc[~mask, "valid_sku"] = True
    payments.loc[mask, "valid_sku"] = False

    data_dict["payments"] = payments

    return data_dict


def get_list_order_costs(data_dict: dict) -> dict:
    """ """
    payments = data_dict["payments"]
    mask = (payments["valid_sku"] == True) & (payments["valid_orderid"] == True)
    df = payments[mask].copy()
    unit_costs = list(set(df["amount-description"]))
    exclusions = ["Tax", "tax", "GST", "SGST", "IGST", "CGST", "TDS", "TCS"]
    unit_tax_costs = [cost for cost in unit_costs if any(e in cost for e in exclusions)]
    unit_costs_non_tax = [
        cost for cost in unit_costs if not any(e in cost for e in exclusions)
    ]

    data_dict["unit_costs"] = unit_costs_non_tax
    data_dict["unit_tax_costs"] = unit_tax_costs

    return data_dict


def create_monthly_orders_and_estimates(
    data_dict: dict,
    status: tuple,
) -> dict:
    """ """

    payments = data_dict["payments"]
    mtr = data_dict["mtr"]
    payment_status = status[0]
    mtr_status = status[1]

    mask = payments["transaction-type"] == payment_status
    order_payments = payments[mask].copy()

    payment_pivot = pd.pivot_table(
        order_payments, values="amount", index="key", columns="amount-description"
    )
    cost_cols = payment_pivot.columns
    payment_pivot = payment_pivot.reset_index()

    unit_costs_dict = {}

    for m in set(mtr["month_year"]):
        mtr_order_payment = mtr[
            (mtr["month_year"] == m) & (mtr["transaction type"] == mtr_status)
        ].copy()
        req_cols = ["key", "month_year", "month_year_end"]
        drop_cols = [col for col in mtr_order_payment.columns if col not in req_cols]
        mtr_order_payment.drop(columns=drop_cols, inplace=True)
        mtr_order_payment = mtr_order_payment.merge(
            payment_pivot, on="key", how="left", indicator=True
        )

        total_orders = mtr_order_payment.index.size
        mask = mtr_order_payment["_merge"] == "both"
        settled_orders = mtr_order_payment[mask].index.size

        for col in cost_cols:
            mtr_order_payment[col] = (
                mtr_order_payment[col] * total_orders / settled_orders
            )

        unit_costs_dict[m] = mtr_order_payment

    data_dict["monthly_unit_costs"] = unit_costs_dict

    return data_dict


def add_monthly_unit_costs_to_mis(data_dict: dict) -> dict:
    """ """

    mis = data_dict["mis"]
    monthly_unit_costs = data_dict["monthly_unit_costs"]
    unit_costs = data_dict["unit_costs"]
    mis = data_dict["mis"]

    dd = defaultdict(dict)
    df = pd.DataFrame()

    for m in monthly_unit_costs:
        dd[m] = {
            col: monthly_unit_costs.get(m)[col].sum()
            for col in monthly_unit_costs.get(m).columns
            if col in unit_costs
        }

    df = pd.DataFrame(dd)
    df.columns = pd.to_datetime(df.columns).date
    mis.columns = pd.to_datetime(mis.columns).date

    mis = mis.add(df, fill_value=0)
    mis = mis.fillna(0)
    for col in mis.columns:
        mis[col] = mis[col].astype(float)

    data_dict["mis"] = mis
    data_dict["unit_costs_mis_format"] = df

    return data_dict


def get_list_overhead_costs(data_dict: dict) -> dict:
    """ """
    payments = data_dict["payments"]
    unit_costs = data_dict["unit_costs"]
    unit_tax_costs = data_dict["unit_tax_costs"]

    mask = (~payments["amount-description"].isin(unit_costs)) & (
        ~payments["amount-description"].isin(unit_tax_costs)
    )
    df = payments[mask].copy()
    overhead_costs = list(set(df["amount-description"]))
    tax_exclusions = ["Tax", "tax", "GST", "SGST", "IGST", "CGST", "TDS", "TCS"]
    other_exclusions = ["TransactionTotalAmount"]
    exclusions = tax_exclusions + other_exclusions
    overhead_tax_costs = [
        cost
        for cost in overhead_costs
        if cost is not None and any(e in cost for e in tax_exclusions)
    ]
    overhead_costs_non_tax = [
        cost
        for cost in overhead_costs
        if cost is not None and not any(e in cost for e in exclusions)
    ]

    data_dict["overhead_costs"] = overhead_costs_non_tax
    data_dict["overhead_tax_costs"] = overhead_tax_costs

    return data_dict


def add_overheads_to_mis(data_dict: dict) -> dict:
    """ """

    mis = data_dict["mis"]
    payments = data_dict["payments"]
    overhead_costs = data_dict["overhead_costs"]

    mask = payments["amount-description"].isin(overhead_costs)
    payments = payments[mask].copy()
    overhead_df = pd.pivot_table(
        payments,
        index="amount-description",
        columns="month_year",
        values="amount",
        aggfunc="sum",
    )

    mis = mis.add(overhead_df, fill_value=0)

    data_dict["mis"] = mis
    data_dict["overheads"] = overhead_df

    return data_dict


def add_taxes(data_dict: dict) -> dict:

    unit_tax_costs = data_dict["unit_tax_costs"]
    overhead_tax_costs = data_dict["overhead_tax_costs"]
    mis = data_dict["mis"]
    payments = data_dict["payments"]

    exclusions = [
        "Product Tax",
        "Product tax discount",
        "TDS Reimbursement",
        "TDS (Section 194-O)",
    ]

    tax_cols_all = unit_tax_costs + overhead_tax_costs
    tax_cols = [
        t for t in tax_cols_all if t is not None and not any(e in t for e in exclusions)
    ]
    tds_cols = ["TDS Reimbursement", "TDS (Section 194-O)"]
    sales_tax_cols = ["Product Tax", "Product tax discount"]

    mask = payments["amount-description"].isin(tax_cols)
    df = payments[mask].copy()
    fee_tax_all = pd.pivot_table(
        df,
        index="amount-description",
        values="amount",
        columns="month_year",
        aggfunc="sum",
    )
    fee_tax_mis = pd.DataFrame([fee_tax_all.sum()], index=["tax_on_fees"])

    mask = payments["amount-description"].isin(sales_tax_cols)
    df = payments[mask].copy()
    gst_tax_all = pd.pivot_table(
        df,
        index="amount-description",
        values="amount",
        columns="month_year",
        aggfunc="sum",
    )
    gst_tax_mis = pd.DataFrame([gst_tax_all.sum()], index=["gst"])
    gst_tax_mis = gst_tax_mis * (-1)

    mask = payments["amount-description"].isin(tds_cols)
    df = payments[mask].copy()
    tds_all = pd.pivot_table(
        df,
        index="amount-description",
        values="amount",
        columns="month_year",
        aggfunc="sum",
    )
    tds_mis = pd.DataFrame([tds_all.sum()], index=["tds"])

    mis = mis.add(gst_tax_mis, fill_value=0)
    mis = mis.add(fee_tax_mis, fill_value=0)
    mis = mis.add(tds_mis, fill_value=0)
    mis = mis.fillna(0)

    data_dict["mis"] = mis
    data_dict["sales_tax_break_up"] = gst_tax_all
    data_dict["fee_tax_break_up"] = fee_tax_all
    data_dict["tds_break_up"] = tds_all

    return data_dict


def create_amazon_india_mis(_path_: str = "/Users/ap/Desktop/mis/amazon") -> dict:
    """
    Creates Amazon MIS

    """

    print(f"Ingesting files....")
    data_dict = ingest_files(_path_)
    data_dict = tag_current_month_orders(data_dict)
    data_dict = add_gross_sales(data_dict)
    data_dict = add_gross_quantity(data_dict)
    data_dict = create_monthly_payments(data_dict)
    data_dict = map_sku_to_asin(data_dict)
    data_dict = create_keys(data_dict)
    data_dict = add_standard_statuses(data_dict)

    ## Create dataframes for gross sales, returns, replacements and cancellations
    order_statuses = [
        "Net Delivered",
        "Return + RTO",
        "Cancelled",
        "Replacements",
        "Gross Sale",
    ]

    print(f"Mapping unit costs....")
    for status in order_statuses:
        data_dict = create_mtr_pivots(data_dict, "gross_quantity_mis", status)
        data_dict = create_mtr_pivots(data_dict, "gross_sale_mis", status)

    data_dict = create_mis_structure(data_dict)
    data_dict = tag_sku_and_orders(data_dict)
    data_dict = break_payments_by_month(data_dict)
    data_dict = get_list_order_costs(data_dict)
    data_dict = get_list_overhead_costs(data_dict)

    status_tuples = [("Order", "Net Delivered"), ("Refund", "Return + RTO")]
    for status in status_tuples:
        data_dict = create_monthly_orders_and_estimates(data_dict, status)
        data_dict = add_monthly_unit_costs_to_mis(data_dict)

    print(f"Add overhead costs....")
    data_dict = add_overheads_to_mis(data_dict)
    print(f"Add tax cols....")
    data_dict = add_taxes(data_dict)
    print(f"Code successfully executed.")

    print(f"MIS:\n{data_dict['mis']}")

    return data_dict
