import pandas as pd
import numpy as np
import sys
import re
from typing import Optional
from datetime import timedelta, datetime, date
import calendar
from collections import defaultdict
import warnings


def summarize_by_transaction_files(
    df_transactions: pd.DataFrame,
    cogs_df: pd.DataFrame,
    marketing_df: pd.DataFrame,
    exclusions: list = ["Service Fees"],
    summary_level: str = "M",
) -> pd.DataFrame:
    """
    This function summarizes the transaction file obtained from Amazon into a very high level summary. It calcuated CM1, CM2 and CM3 at a level specified by the user
    Parameters:
    (1) df_transactions: a single dataframe of transaction files obtained from Amazon. This is required
    (2) cogs_df: a dataframe that maps COGS for each order ID. This is created from "calculate_cogs_by_orderid" function.
    (3) marketing_df: a dataframe containing total marketing spends by channel. This could be at a daily level, campaign level. Logics will have to be built to get the right marketing amount. Should have atleast 'date' and 'spend' column
    (4) exclusions (default value = 'Service Fees'). If marketing if taken from the marketing files, then this should be default, else change to empty list
    (5) summary_level(default value = 'M') same parameters used in to_period (Y,Q,M,W,D,H,T,min,S) - Year, Quarter, Month etc.. (T and min are same)

    RIP Ozzy (03-Dec-1948 to 22-Jul-2025)

    """
    df = df_transactions[~df_transactions["Transaction type"].isin(exclusions)].copy()
    df["summary_level"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period(
        summary_level
    )
    df.columns = df.columns.astype(str).str.lower().str.strip()
    marketing_df.columns = marketing_df.columns.astype(str).str.lower().str.strip()
    marketing_df["summary_level"] = pd.to_datetime(
        marketing_df["date"], errors="coerce"
    ).dt.to_period(summary_level)
    summary_levels = df["summary_level"].unique()
    summary_dict = {}
    summary_dict["Index"] = [
        "Sales",
        "Promotions",
        "Net Sales",
        "COGS",
        "CM1",
        "CM1 %age",
        "Transaction fees",
        "Transaction fees %age",
        "Refunds",
        "Overheads",
        "CM2",
        "CM2%age",
        "Marketing",
        "CM3",
        "CM3 %age",
    ]

    for level in summary_levels:
        cogs_dict = cogs_df.to_dict()
        mkt_df = marketing_df[marketing_df["summary_level"] == level].copy()
        cond = df["transaction type"] == "Order Payment"

        ## Get settled sales
        sales = df[cond]["total product charges"].sum()
        promotions = df[cond]["total promotional rebates"].sum()
        net_sales = sales + promotions

        ## Get CM1
        df = df.assign(
            cogs=lambda x: x["order id"].map(cogs_dict).fillna(0)
            * x["transaction type"].map({"Refund": 1, "Order Payment": -1})
        )
        cogs = df[df["transaction type"] == "Order Payment"]["cogs"].sum()
        cm1 = net_sales + cogs
        cm1_pct = cm1 / net_sales

        ## Get CM2
        transaction_fees = df[cond]["amazon fees"].sum()
        transaction_fees_pct = (transaction_fees / net_sales) * (-1)
        refunds = (
            df[df["transaction type"] == "Refund"]
            .agg({"total (usd)": "sum", "cogs": "sum"})
            .sum()
        )
        all_transaction_types = list(df["transaction type"].unique())
        all_transaction_types.remove("Order Payment")
        all_transaction_types.remove("Refund")
        overhead_types = all_transaction_types
        cond2 = df["transaction type"].isin(overhead_types)
        overheads = df[cond]["other"].sum() + df[cond2]["total (usd)"].sum()
        cm2 = cm1 + transaction_fees + refunds + overheads
        cm2_pct = cm2 / net_sales

        ## Get CM3
        marketing = mkt_df["spend"].sum() * (-1)
        cm3 = cm2 + marketing
        cm3_pct = cm3 / net_sales

        col_vals = [
            sales,
            promotions,
            net_sales,
            cogs,
            cm1,
            cm1_pct,
            transaction_fees,
            transaction_fees_pct,
            refunds,
            overheads,
            cm2,
            cm2_pct,
            marketing,
            cm3,
            cm3_pct,
        ]
        summary_dict[level] = col_vals

    output_df = pd.DataFrame(summary_dict)
    return output_df


def calculate_cogs_by_orderid(
    cogs_mapper: pd.DataFrame, order_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function calculated COGS by orderID. It is required as an input function to "summarize_by_transaction_files" function
    Parameters:
    (a) cogs_mapper: a DataFrame containing ASIN level cogs (as this is specific to Amazon)
    (b) order_df: A DataFrame of Amazon order file

    RIP Christopher Lee (27-May-1922 to 07-Jun-2015)
    """

    cogs_mapper.columns = cogs_mapper.columns.astype(str).str.lower().str.strip()
    cogs_dict = cogs_mapper.set_index("asin")["cogs"].to_dict()
    order_df = order_df.assign(
        cogs_per_unit=lambda x: x["asin"].map(cogs_dict),
        cogs=lambda x: x["cogs_per_unit"] * x["quantity"],
    )
    cogs_df = order_df.groupby("amazon-order-id")["cogs"].sum()
    return cogs_df


def add_date_to_mtr_file(order_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a date column to the MTR dataframe.
    **** IMPORTANT: While consolidating MTR files, we need to get the month of the file from the name. This should be stored in a column called "mtr_date" and should be in 01-mm-yy format.
    The function then creates final_date column, where if the invoice_date column of the MTR file is not from the same period as the mtr_date, then final_date should be mtr_date, else
    final_date shold be equal to invoice_date.
    This is important because we need to
    """
    try:
        order_df["mtr_date"] = pd.to_datetime(order_df["mtr_date"], errors="coerce")
        order_df["Invoice Date"] = pd.to_datetime(
            order_df["Invoice Date"], errors="coerce"
        ).dt.tz_convert("Asin/Kolkata")
        order_df = order_df.assign(
            mtr_date_period=lambda df: df["mtr_date"].dt.to_period("M"),
            invoice_date_period=lambda df: df["Invoice Date"]
            .dt.tz_convert(None)
            .dt.to_period("M"),
            final_date=lambda df: df.apply(
                lambda row: (
                    row["Invoice Date"]
                    if row["invoice_date_period"] == row["mtr_date_period"]
                    else row["mtr_date"]
                )
            ),
        ).drop(columns=["mtr_date_period", "invoice_date_period"])
    except KeyError:
        warnings.warn(
            f"ERROR: mtr_date column not added to individual MTR files. Please add the columns manually"
        )
        print(
            f"**** IMPORTANT: While consolidating MTR files, we need to get the month of the file from the filename. This should be stored in a column called 'mtr_date' and should be in 01-mm-yy format."
        )
        sys.exit(1)

    return order_df


def clean_orders(
    order_df: pd.DataFrame, mtr_file_bool: bool, incl_tax: bool
) -> pd.DataFrame:
    """
    This function cleans the orders file to ensure data is assimilated correctly. It accepts either an MTR fle input or an FBA all orders file as input and does the following:
    a. Aggregates the data by a "key" (order-id X asin) using a custom aggregation function
    b. creates a list of status changes
    c. creates a new column called "month" which is a month period identifier
    Arguments:
        a. order_df: a dataframe or either MTR files of orders.txt file
        b. mtr_file_bool: a Boolean input. Use False for FBA all orders file (default value) and True for MTR files
        c. incl_tax: A Boolean input used to determine if the output is required with or without tax
    """

    if not incl_tax and mtr_file_bool:
        order_df["Invoice Amount"] = order_df["Tax Exclusive Gross"]
    elif not incl_tax and not mtr_file_bool:
        order_df["item-price"] = order_df["item-price"].fillna(0)
        order_df["item-tax"] = order_df["item-tax"].fillna(0)
        order_df["item-price"] = order_df["item-price"] + order_df["item-tax"]

    if not mtr_file_bool:
        pattern = "Non-Amazon"
        non_amazon_sales_mask = order_df["sales-channel"].astype(str).str.match(pattern)
        order_df = order_df[~non_amazon_sales_mask]

    cols_dict = {
        True: [
            "Order id",
            "final_date",
            "Transaction Type",
            "Fulfillment Channel",
            "Sku",
            "Asin",
            "Quantity",
            "Principal Amount",
            "Ship To City",
            "Ship To State",
        ],
        False: [
            "amazon-order-id",
            "purchase-date",
            "order-status",
            "item-status",
            "fulfillment-channel",
            "sales-channel",
            "order-channel",
            "sku",
            "asin",
            "quantity",
            "item-price",
            "ship-city",
            "ship-state",
        ],
    }

    rename_dict = {
        True: {
            "Order id": "order-id",
            "final_date": "date",
            "Transaction Type": "status",
            "Fulfillment Channel": "fulfillment-channel",
            "Sku": "sku",
            "Asin": "asin",
            "Quantity": "quantity",
            "Invoice Amount": "sale",
            "Ship To City": "city",
            "Ship To State": "state",
        },
        False: {
            "amazon-order-id": "order-id",
            "purchase-date": "date",
            "order-status": "status",
            "item-status": "item-status",
            "fulfillment-channel": "fulfillment-channel",
            "sales-channel": "sales-channel",
            "order-channel": "order-channel",
            "sku": "sku",
            "asin": "asin",
            "quantity": "quantity",
            "item-price": "sale",
            "ship-city": "city",
            "ship-state": "state",
        },
    }

    numeric_cols_dict = {True: ["quantity", "sale"], False: ["quantity", "sale"]}

    list_cols_dict = {True: ["status", "item-status"], False: ["status", "item-status"]}

    req_cols = cols_dict.get(mtr_file_bool)
    if mtr_file_bool:
        order_df = add_date_to_mtr_file(order_df)

    df = order_df[req_cols].copy()
    df = df.rename(columns=rename_dict.get(mtr_file_bool))
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_convert(
        "Asia/Kolkata"
    )
    df["date"] = pd.to_datetime(df["date"])

    if mtr_file_bool:
        df["item-status"] = pd.NA

    numeric_cols = numeric_cols_dict.get(mtr_file_bool)
    list_cols = list_cols_dict.get(mtr_file_bool)

    agg_func = {
        k: "first"
        for k in rename_dict.get(mtr_file_bool).values()
        if k not in numeric_cols and k not in list_cols
    }
    for k in rename_dict.get(mtr_file_bool).values():
        if k in numeric_cols:
            agg_func[k] = "sum"
        elif k in list_cols:
            agg_func[k] = list

    df = (
        df.assign(
            key=lambda df: df["order-id"].astype(str) + df["asin"].astype(str),
            counts=lambda df: df["key"].map(df["key"].value_counts()),
        )
        .groupby("key", as_index=False)
        .agg(agg_func)
        .assign(status_changes=lambda df: df["status"] + df["item-status"])
    )
    df["month"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.to_period("M")
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["date"] = df["date"].dt.date

    return df


def clean_payments(payment_df: pd.DataFrame, amt_desc_to_remove: list) -> pd.DataFrame:
    """
    This function does the following:
    a. creates a new column called "month" which is a month period identifier
    b. creates a new column "classification" which will be used to identify the allocation methodology of the cost head (amount-description)
    Arguments:
        payment_df: the settlement file from Amazon
        amt_desc_to_remove: a list of amount-descriptions that have to be removed from payments_df
    """
    cols_to_fill = ["settlement-start-date", "settlement-end-date", "settlement-id"]
    # force_overheads = ['Debt adjustment against COD Transactions and Non-Transactional Fee Accounts', 'Debt adjustment against Electronic Transaction (Credit Card/Net Banking/GC) Accounts', 'Order Cancellation Charge', 'MiscAdjustment', 'Payable to Amazon']
    # payment_df.loc[payment_df['amount-description'].isin(force_overheads), 'sku'] = "NA"
    payment_df[cols_to_fill] = payment_df[cols_to_fill].ffill()

    payment_df["date"] = pd.to_datetime(
        payment_df["settlement-start-date"], errors="coerce", dayfirst=True
    ).dt.tz_convert("Asia/Kolkata")

    payment_df["month"] = payment_df["date"].dt.tz_convert(None).dt.to_period("M")
    payment_df["amount"] = payment_df["amount"].astype(float)

    ## within payments consol, identify valid oid
    pattern1 = r"\d{3}-\d{7}-\d{7}"
    pattern2 = r"s\d{2}-\d{7}-\d{7}"

    oid_match = payment_df["order-id"].astype(str).str.match(pattern1) | payment_df[
        "order-id"
    ].astype(str).str.match(pattern2)
    sku_match = (payment_df["sku"].isna()) | (payment_df["sku"] == "NA")
    payment_df = payment_df.assign(is_valid_oid=oid_match, is_valid_sku=~sku_match)

    payment_df["classification"] = "all_cost"
    payment_df.loc[oid_match, "classification"] = "order_cost"
    payment_df.loc[~sku_match, "classification"] = "sku_cost"
    payment_df.loc[oid_match & ~sku_match, "classification"] = "unit_cost"
    payment_df = payment_df[
        ~payment_df["amount-description"].isin(amt_desc_to_remove)
    ].copy()
    payment_df.rename(columns={"quantity-purchased": "quantity"}, inplace=True)

    ## FORCEFULLY CHANGE ORDER CANCELLATION CHARGE TO all_cost
    force_charge_list = ['Order Cancellation Charge', 'OrderCancellationChargeCGST', 'OrderCancellationChargeSGST', 'OrderCancellationChargeIGST']
    payment_df.loc[
        payment_df["amount-description"].isin(force_charge_list),
        "classification",
    ] = "all_cost"

    return payment_df


def clean_cogs(df: pd.DataFrame()) -> pd.DataFrame:
    """
    This function cleans and returns a cogs dict of the format key1 = applicable month, values1 = a dictionary with keys2 = skuid and values2=cogs
    """
    df.columns = df.columns.str.lower().str.strip()
    df["applicable month"] = pd.to_datetime(df["applicable month"], errors="coerce")
    df["month"] = df["applicable month"].dt.to_period("M")
    req = ["asin", "cogs", "month"]
    df = df[req].copy()

    return df


def calculate_net_delivered_quantity(
    order_df: pd.DataFrame, return_df: pd.DataFrame, replacement_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function calculates the net_delivered_quantity for each order
    arguments:
    a. order_df or mapped_orders file
    b. return_df: a Dataframe of all returns from amazon
    c. replacement_df: A Dataframe of all replacement from amazon
    """
    return_df["key"] = return_df["order-id"].astype(str) + return_df["asin"].astype(str)
    num_cols = ["quantity"]
    agg_func = {
        k: ("sum" if k in num_cols else "first")
        for k in return_df.columns
        if k != "key"
    }
    return_df = return_df.groupby("key", as_index=False).agg(agg_func)

    replacement_df["key"] = replacement_df["original-amazon-order-id"].astype(
        str
    ) + replacement_df["asin"].astype(str)
    num_cols = ["quantity"]
    agg_func = {
        k: ("sum" if k in num_cols else "first")
        for k in replacement_df.columns
        if k != "key"
    }
    replacement_df = replacement_df.groupby("key", as_index=False).agg(agg_func)

    order_df = order_df.merge(return_df, on="key", how="left", suffixes=["", "_ret"])
    order_df = order_df.merge(
        replacement_df, on="key", how="left", suffixes=["", "_rep"]
    )
    order_df["quantity_ret"] = order_df["quantity_ret"].fillna(0)
    order_df["quantity_rep"] = order_df["quantity_rep"].fillna(0)
    order_df["net_delivered_qty"] = (
        order_df["quantity"] + order_df["quantity_ret"] + order_df["quantity_rep"]
    )

    return order_df


def map_marketing(
    mapped_orders: pd.DataFrame, daily_marketing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function allocates marketing spend to each row of the mapped orders file
    arguments:
    a. mapped_orders or order_df
    b. daily_marketing_df: A dataframe containing only two columns - date and marketing spend.
    can expand this function in the future to allocate based on different logics
    """

    daily_marketing_df.columns = (
        daily_marketing_df.columns.astype(str).str.lower().str.strip()
    )
    min_date = mapped_orders["date"].min()
    max_date = mapped_orders["date"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    daily_marketing_df["date"] = pd.to_datetime(
        daily_marketing_df["date"], errors="coerce"
    ).dt.date

    ## Check for days when there is a marketing spend, but no sales
    missing_days = [
        d
        for d in all_dates
        if d not in set(mapped_orders["date"]) and d in set(daily_marketing_df["date"])
    ]

    new_rows = {"date": missing_days}

    cols = [col for col in mapped_orders.columns if col != "date"]
    for col in cols:
        new_rows[col] = pd.NA

    new_rows = pd.DataFrame(new_rows)
    mapped_orders = pd.concat([mapped_orders, new_rows], ignore_index=True)
    daily_marketing_df.columns = daily_marketing_df.columns.str.lower().str.strip()
    daily_marketing_df["month"] = pd.to_datetime(
        daily_marketing_df["date"], errors="coerce", dayfirst=True
    ).dt.to_period("M")
    mapped_orders["daily_order_count"] = mapped_orders["date"].map(
        mapped_orders.groupby("date")["net_delivered_qty"].sum()
    )
    daily_marketing_dict = daily_marketing_df.groupby("date")["spend"].sum()
    daily_marketing_dict = {
        pd.to_datetime(k, errors="coerce"): v for k, v in daily_marketing_dict.items()
    }
    mapped_orders["total_marketing_spend"] = mapped_orders["date"].map(
        daily_marketing_dict
    )
    mapped_orders["marketing"] = (
        (mapped_orders["total_marketing_spend"] / mapped_orders["daily_order_count"])
        * mapped_orders["net_delivered_qty"]
    ).where(mapped_orders["daily_order_count"] != 0, 0)

    return mapped_orders


def map_overhead_payments(
    mapped_orders: pd.DataFrame,
    payment_df: pd.DataFrame,
    months: pd.PeriodIndex,
    pure_all_overheads: set,
    overlap_cols: set,
) -> pd.DataFrame:
    """
    This function creates a dictionary that holds values for overheads. There are 5 cases that can happen when mapping payments to orders (keys)
    Case1: Payment file has both orderid and sku information and hence a key can be created - simplest case, a simple merge with pivoted payment table and orders will work
    case2: Payment file has only order information - here we allocate the cost only to those orders where the order id matches. We need to ensure that all the 'amount-descriptions'
           in this case are unique (are not repeated, else we will get erroreous results)
    Case3: Payment file has only sku information - we allocate them over the entire orders - basically treat them as simple overheads
    Case4: Neither order id nor sku is present, another simple case, we allocate them equally to all the rows

    This function only deals with Case3 and Case4 as the others are simple and straight forward and have been dealt with in the main body
    Here the complication is in isolating amount-descriptions
    The function below creates a nested dictionary, with months as primary keys and ['orders', 'payments', 'mapped_orders_overheads', 'mapped_orders_overlaps'] as secondary keys.
    We can then extract relevant mapped_orders to merge with the original mapped orders to get payment recon file.

    Arguments:
        a. mapped_orders: A Dataframe where Case1 and Case2has already been mapped
        b. payment_df = A consolidated dataframe containing the Payment files
        b. months: A list of months for which the mapping needs to be done
        c. pure_all_overheads: A set of 'amount-descriptions' where both orderid's and sku information is missing, and which do not fall under any other types
        d. overlap_cols: A set of 'amount-descriptions' which fall under multiple Cases. Ideally, this should be a null set
    """

    monthly_pairs = defaultdict(dict)

    for m in months.values:
        orders_m = mapped_orders[mapped_orders["month"] == m].copy()
        payments_m = payment_df[payment_df["month"] == m].copy()

        monthly_pairs[m]["orders"] = orders_m
        monthly_pairs[m]["payments"] = payments_m
        qty_for_month = orders_m["net_delivered_qty"].sum()
        keys_for_month = orders_m["key"]
        qty_array = np.array(orders_m["net_delivered_qty"])
        payments_m["amount_per_qty"] = np.where(
            qty_for_month != 0, payments_m["amount"] / qty_for_month, 0
        )

        overhead_col_sums = {}
        for k in pure_all_overheads:
            total_amount = payments_m.loc[
                (payments_m["amount-description"] == k)
                & (payments_m["classification"] == "all_cost"),
                "amount",
            ].sum()
            amount_per_unit = total_amount / qty_for_month if qty_for_month else 0
            overhead_col_sums[k] = (amount_per_unit * qty_array).tolist()

        overhead_col_sums["key"] = keys_for_month
        overhead_col_sums["classification"] = "all_cost"
        pure_all_overheads_df = pd.DataFrame(data=overhead_col_sums)
        pure_all_overheads_df["month"] = m
        monthly_pairs[m]["mapped_orders_overheads"] = pure_all_overheads_df

        overlap_cols_sums = {}
        for k in overlap_cols:
            total_amount = payments_m.loc[
                (payments_m["amount-description"] == k)
                & (payments_m["classification"] == "sku_cost"),
                "amount",
            ].sum()
            per_unit_amount = total_amount / qty_for_month if qty_for_month else 0
            overlap_cols_sums[k] = (per_unit_amount * qty_array).tolist()

        overlap_cols_sums["key"] = keys_for_month
        overlap_cols_sums["classification"] = "sku_cost"
        overlap_overheads_df = pd.DataFrame(data=overlap_cols_sums)
        overlap_overheads_df["month"] = m
        monthly_pairs[m]["mapped_orders_overlaps"] = overlap_overheads_df

    return monthly_pairs


def remove_tax_cols(payment_df: pd.DataFrame):
    """
    This function removes tax columns from payment file.
    """

    all_payment_types = list(set(payment_df["amount-description"]))
    tax_identifiers = ["gst", "tax"]
    non_tax_cols = [
        col
        for col in all_payment_types
        if not any(tax_id in col for tax_id in tax_identifiers)
    ]
    payment_df = payment_df[payment_df["amount-description"].isin(non_tax_cols)]

    return payment_df


def add_unsettled_fees(mapped_orders: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds estimates fees for unsettled orders
    Arguments:
    a. mapped_orders: Fully prepared mapped_orders file (A Dataframe)
    b. unit_costs: a list of unit costs only (costs that are mapped to both orderid and sku)
    """

    settled_orders = mapped_orders[mapped_orders["_merge"] == "both"].copy()
    months = mapped_orders["month"].unique()

    average_cols = [
        "FBA Weight Handling Fee",
        "Fixed closing fee",
        "Technology Fee",
        "FBA Pick & Pack Fee",
        "TechnologyFee",
        "FBAPerUnitFulfillmentFee",
    ]
    pct_cols = ["Commission"]
    average_fee_dict = defaultdict(dict)
    pct_fee_dict = defaultdict(dict)

    ### Get estimates as a dict
    for m in months:
        df = settled_orders[settled_orders["month"] == m].copy()
        for col in average_cols:
            average_fee_dict[m][col] = np.where(
                df[col].sum() / df["net_delivered_qty"].sum(),
                df["net_delivered_qty"] != 0,
                0,
            )
        for asin in df["asin"].unique():
            pct_fee_dict[m][asin] = np.where(
                df[df["asin"] == asin]["Commission"].sum()
                / df[df["asin"] == asin]["settled_sales"].sum(),
                df[df["asin"] == asin]["settled_sales"] != 0,
                0,
            )

    ### Map the estimated fees to the MIS
    for m in months:
        for col in average_cols:
            mapped_orders[col] = np.where(
                (mapped_orders["_merge"] == "left_only"),
                average_fee_dict.get(col) * mapped_orders["net_delivered_qty"],
                mapped_orders[col],
            )
        for asin in mapped_orders["asin"].unique():
            mask = (mapped_orders["asin"] == asin) & (
                mapped_orders["_merge"] == "left_only"
            )
            mapped_orders["Commission"] = np.where(
                mask,
                pct_fee_dict.get(asin) * mapped_orders["settled_sales"],
                mapped_orders["Commission"],
            )

    return mapped_orders


def remove_common_cols(df, str_to_check):
    col_dict = defaultdict()
    for col in df.columns.astype(str):
        if col + str_to_check in df.columns:
            col_dict[col] = col + str_to_check

    for col in col_dict:
        col2 = col + str_to_check
        df[col] += df[col2].fillna(0)
        df.drop(columns=col2, inplace=True)

    return df


def map_amazon_payments(
    order_df: pd.DataFrame,
    payment_df: pd.DataFrame,
    return_df: pd.DataFrame,
    replacement_df: pd.DataFrame,
    daily_marketing_df: pd.DataFrame,
    cogs_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    mtr_file_bool: bool = False,
    incl_tax: bool = True,
) -> pd.DataFrame:
    """
    This function accepts both orders.txt (amazon all order) or the MTR file and maps the payment against each order_id using the payments files.
    Arguments:
        a. order_df: A DataFrame prepared using the clean_orders function described earlier. Orders could be from order file or the mtr file
        b. payments_df: Consolidated payments DataFrame
        c. return_df: A DataFrame with FBA returns
        d. replacement_df: A Dataframe with FBA replacements
        e. daily_marketing_df: A Dataframe with daily marketing spends. Note, this file will have only two columns - date and spend
        e. cogs_df: A dataframe containing monthwise COGS for each sku
        f. start_date: The date from which to start the reconciliation
        g. end_date: The date until which to do the reconciliation
        h. mkt: Boolean flag to determine whether we should use the marketing amount from payments or a separate marketing file will be provided. False = use marketing cost from payments file. Default=True
        i. incl_tax: Boolen flag to calculate this pre-tax or post tax. Default=True - include taxes
    Note: Dates can be obtained from the order file, but this way we can get the reconciliation for a specific month if required
    """

    print(f"{'*' * 250}\n")
    print(
        f"Required files:\norder_df (Amazon orders),\npayment_df (Consolidated payments),\nreturn_df (Returns),\nreplacement_df (Replacements),\ndaily_marketing_df (Marketing),\ncogs_df (COGS),\nstart_date (Optional),\nend_date (Optional),\nmtr_file_bool (Default: False),\nincl_tax (Default: False)"
    )
    print(f"{'*' * 250}\n")

    warnings.filterwarnings("once")

    ## check if all files are present
    err_counter = 0
    missing_files = []

    if order_df is None:
        err_counter += 1
        missing_files.extend(["order_df"])

    if payment_df is None:
        err_counter += 1
        missing_files.extend(["payment_df"])

    if return_df is None:
        err_counter += 1
        missing_files.extend(["return_df"])

    if replacement_df is None:
        err_counter += 1
        missing_files.extend(["replacement_df"])

    if daily_marketing_df is None:
        err_counter += 1
        missing_files.extend(["daily_marketing_df"])

    if cogs_df is None:
        err_counter += 1
        missing_files.extend(["cogs_df"])

    if err_counter > 0:
        warnings.warn(f"Some Input files are missing. See Below\n")
        for i in range(err_counter):
            warnings.warn(f"{missing_files[i]}\n")

    mkt = True  ## Setting this to True - need to check the logic for when marketing file is unavialble. In the meantime - create a marketing_df file using "TransactionTotalAmount" from payments file and distributing it by days
    warnings.warn(
        "\nWARNING: mkt has been set to True by default. \nIn case marketing file is unavailable, create one using data from payments file (TransactionTotalAmount column) and distribute it by days.\n IT IS IMPORTANT TO DISTRIBUTE THE AMOUNTS BY DAYS\n"
    )
    print(f"Executing code....")

    GARBAGE_DESCRIPTIONS = []
    if mkt:
        GARBAGE_DESCRIPTIONS.append("TransactionTotalAmount")

    if not incl_tax:
        amt_desc = set(payment_df["amount-description"])
        tax_descriptors = ["gst", "tax", "tds"]
        tax_cols = [
            desc for desc in amt_desc if any(t in desc.lower() for t in tax_descriptors)
        ]
        GARBAGE_DESCRIPTIONS.extend(tax_cols)

    print(f"Cleaning orders...")
    order_df = clean_orders(order_df, mtr_file_bool, incl_tax)
    print(f"Orders cleaned. Cleaning payments...")
    payment_df = clean_payments(payment_df, GARBAGE_DESCRIPTIONS)
    print(f"Data cleaned.")
    cogs_df = clean_cogs(cogs_df)
    print(f"COGS cleaned.")

    if start_date is None:
        start_date_temp = order_df["date"].min()
        start_date = start_date_temp

    if end_date is None:
        end_date_temp = order_df["date"].max()
        end_date = end_date_temp

    start = pd.to_datetime(start_date).to_period("M")
    end = pd.to_datetime(end_date).to_period("M")
    months = pd.period_range(start=start, end=end, freq="M")

    sku_to_asin_mapper = order_df.groupby("sku").agg({"asin": "first"}).to_dict()
    payment_df = payment_df.assign(
        asin=lambda df: df["sku"].map(sku_to_asin_mapper.get("asin")),
        key=lambda df: df["order-id"].astype(str) + df["asin"].astype(str),
    )
    order_df = calculate_net_delivered_quantity(order_df, return_df, replacement_df)
    payment_pivot = pd.pivot_table(
        payment_df[payment_df["classification"] == "unit_cost"],
        columns="amount-description",
        index="key",
        values="amount",
        aggfunc="sum",
    )
    payment_pivot = payment_pivot.reset_index()
    payment_pivot.dropna(axis=1, how="all", inplace=True)
    mapped_orders = order_df.merge(
        payment_pivot, on="key", how="left", indicator=True
    )  ## Map orderid X SKU level payments
    unit_costs = [col for col in payment_pivot.columns if col != "key"]
    mapped_orders[unit_costs] = mapped_orders[unit_costs].fillna(0)
    # mapped_orders['unit_costs'] = mapped_orders[unit_costs].sum(axis=1)
    print(f"Unit costs mapped.")

    ## Drop ccancelled rows
    mapped_orders = mapped_orders.assign(
        cancel_flag=lambda df: df["status_changes"].apply(
            lambda statuses: any(
                "cancelled" in s.lower() or "canceled" in s.lower()
                for s in statuses
                if isinstance(s, str)
            )
        ),
        no_payment_flag=lambda df: df[unit_costs].sum(axis=1) == 0,
    ).assign(drop_flag=lambda df: df["cancel_flag"] & df["no_payment_flag"])
    mapped_orders = mapped_orders[~mapped_orders["drop_flag"]].copy()
    payment_df["order_counts"] = (
        payment_df["order-id"].map(mapped_orders["order-id"].value_counts()).fillna(0)
    )
    payment_df["amount_per_order"] = np.where(
        payment_df["order_counts"] != 0,
        payment_df["amount"] / payment_df["order_counts"],
        0,
    )
    order_overhead_cols = payment_df[payment_df["classification"] == "order_cost"][
        "amount-description"
    ].unique()
    other_overhead_cols = payment_df[
        payment_df["classification"].isin(["sku_cost", "all_cost"])
    ]["amount-description"].unique()
    overlap_cols = set(order_overhead_cols) & set(
        other_overhead_cols
    )  ## amount descriptions that are there in both order_cost and all_costs - we need to treat them separately
    pure_order_overheads = set(order_overhead_cols) - overlap_cols
    pure_all_overheads = set(other_overhead_cols) - overlap_cols

    pure_order_overheads_df = pd.pivot_table(
        payment_df[
            (payment_df["amount-description"].isin(pure_order_overheads))
            & (payment_df["classification"] == "order_cost")
        ],
        columns="amount-description",
        index="order-id",
        values="amount_per_order",
        aggfunc="first",
    )
    mapped_orders = mapped_orders.merge(
        pure_order_overheads_df, on="order-id", how="left", suffixes=["", "_orders"]
    )  ## Map payments where only orderid is present

    mapped_orders = remove_common_cols(mapped_orders, "_orders")
    mapped_orders[pure_order_overheads_df.columns] = mapped_orders[
        pure_order_overheads_df.columns
    ].fillna(0)
    print(f"Pure overhead costs allocated.")
    monthly_pairs = map_overhead_payments(
        mapped_orders, payment_df, months, pure_all_overheads, overlap_cols
    )
    print(f"Monthly pairs created.")

    all_overheads_df = pd.concat(
        [monthly_pairs[m].get("mapped_orders_overheads") for m in months],
        axis=0,
        ignore_index=True,
    ).fillna(0)

    all_overlap_df = pd.concat(
        [monthly_pairs[m].get("mapped_orders_overlaps") for m in months],
        axis=0,
        ignore_index=True,
    ).fillna(0)

    req_cols = list(pure_all_overheads)
    req_cols.extend(["key"])
    all_overheads_df_for_merge = all_overheads_df[req_cols].copy()
    mapped_orders = mapped_orders.merge(
        all_overheads_df_for_merge, on="key", how="left", suffixes=["", "_overheads"]
    )
    mapped_orders = remove_common_cols(mapped_orders, "_overheads")

    mapped_orders[list(overlap_cols)] = mapped_orders[list(overlap_cols)].add(
        all_overlap_df[all_overlap_df["classification"] == "overlap_cols"][
            list(overlap_cols)
        ],
        fill_value=0,
    )  ## For amount-descriptions which fall in multiple classifications, we add the columns
    mapped_orders[list(overlap_cols)] = mapped_orders[list(overlap_cols)].fillna(0)
    overhead_cols = list(order_overhead_cols) + list(other_overhead_cols)
    codb_cols = list(unit_costs) + list(overhead_cols)
    print(f"Other overheads allocated.")

    mapped_orders = mapped_orders.merge(cogs_df, on=["month", "asin"], how="left")
    mapped_orders.rename(columns={"cogs": "cogs_pu"}, inplace=True)
    mapped_orders["cogs"] = (
        mapped_orders["cogs_pu"] * mapped_orders["net_delivered_qty"] * (-1)
    )
    if not mkt:
        mapped_orders.rename(
            columns={"TransactionTotalAmount": "marketing"}, inplace=True
        )
    else:
        mapped_orders = map_marketing(mapped_orders, daily_marketing_df)

    mapped_orders["sale"] = pd.to_numeric(mapped_orders["sale"], errors="coerce")

    for col in unit_costs:
        mapped_orders[col] = pd.to_numeric(mapped_orders[col], errors="coerce")

    for col in overlap_cols:
        mapped_orders[col] = pd.to_numeric(mapped_orders[col], errors="coerce")

    if incl_tax:
        mapped_orders["settled_sales"] = (
            mapped_orders["Principal"] + mapped_orders["Product Tax"]
        )
    else:
        mapped_orders["settled_sales"] = mapped_orders["Principal"]

    mapped_orders["sale"] = mapped_orders["sale"].fillna(0)
    mapped_orders["settled_sales"] = mapped_orders["settled_sales"].fillna(0)
    mapped_orders["total_sale_incl_unsettled"] = np.where(
        mapped_orders["_merge"] == "both",
        mapped_orders["settled_sales"],
        mapped_orders["sale"],
    )
    # mapped_orders['settled_sales'] = mapped_orders['Principal']

    mapped_orders["marketing"] = pd.to_numeric(
        mapped_orders["marketing"], errors="coerce"
    ).fillna(0)
    mapped_orders["cogs"] = pd.to_numeric(
        mapped_orders["cogs"], errors="coerce"
    ).fillna(0)
    mapped_orders["total_sale_incl_unsettled"] = pd.to_numeric(
        mapped_orders["total_sale_incl_unsettled"], errors="coerce"
    ).fillna(0)
    mapped_orders["settled_sales"] = pd.to_numeric(
        mapped_orders["settled_sales"], errors="coerce"
    ).fillna(0)

    if mtr_file_bool:
        mapped_orders["cm1"] = (
            mapped_orders["total_sale_incl_unsettled"] + mapped_orders["cogs"]
        )
        mapped_orders = add_unsettled_fees(mapped_orders)
    else:
        mapped_orders["cm1"] = mapped_orders["settled_sales"] + mapped_orders["cogs"]

    mapped_orders["unit_costs"] = mapped_orders[unit_costs].sum(axis=1)
    mapped_orders["platform_overheads"] = mapped_orders[overhead_cols].sum(axis=1)
    mapped_orders["cm2"] = (
        mapped_orders["cm1"]
        + mapped_orders["unit_costs"]
        + mapped_orders["platform_overheads"]
    )
    mapped_orders["cm3"] = mapped_orders["cm2"] + mapped_orders["marketing"]
    mapped_orders["cogs_pct"] = (mapped_orders["cogs"] / mapped_orders["sale"]).where(
        mapped_orders["sale"] != 0, 0
    )

    print(f"Code completed!")

    return mapped_orders, payment_df, all_overheads_df, pure_all_overheads
