import pandas as pd
import numpy as np


def map_payments(
    df_orders: pd.DataFrame,
    df_payments: pd.DataFrame,
    cogs_dict: dict,
    marketing_dict: dict,
) -> pd.DataFrame:
    """
    This function takes meesho order file and maps it to the meesho payments file
    Arguments:
    1) df_orders: Meesho order file
    2) df_payments: Meesho payment file
    3) cogs_dict: A dictionary that maps sku_id and cogs
    4) marketing_dict: A dictionary containing datewise marketing spends
    """

    df_orders.columns = df_orders.columns.astype(str).str.strip().str.lower()
    df_orders["order date"] = pd.to_datetime(df_orders["order date"], errors="coerce")
    df_orders["order date"] = df_orders["order date"].dt.date

    df_payments.columns = df_payments.columns.astype(str).str.lower().str.strip()
    df_payments["order date"] = pd.to_datetime(
        df_payments["order date"], errors="coerce"
    )
    df_payments["order date"] = df_payments["order date"].dt.date

    print(f"Dates corrected.")

    ## Group orders file by sub order number
    agg_func = {
        "order date": "first",
        "customer state": "first",
        "product name": "first",
        "sku": "first",
        "size": "first",
        "quantity": "mean",
        "supplier listed price (incl. gst + commission)": "mean",
        "supplier discounted price (incl gst and commision)": "mean",
        "packet id": "first",
    }

    valid_cols = df_orders.columns
    final_agg_dict = {
        k: v for k, v in agg_func.items() if k in valid_cols
    }  ### We need to add a try, except block here to catch erors or missing fields
    grouped_orders = df_orders.groupby("sub order no", as_index=False).agg(
        final_agg_dict
    )
    grouped_orders["sub_order_counts"] = grouped_orders["sub order no"].map(
        grouped_orders["sub order no"].value_counts()
    )
    print(f"Grouped orders created")

    ## Group payments file by sub order number
    all_cols = df_payments.columns
    agg_by_first = [
        "order date",
        "dispatch date",
        "product name",
        "supplier sku",
        "live order status",
        "product gst %",
        "transaction id",
        "payment date",
        "price type",
        "tds rate %",
        "claims reason",
        "recovery reason",
        "compensation reason",
    ]

    agg_by_mean = ["listing price (incl. taxes)", "quantity"]
    agg_by_cust = [
        "total sale return amount (incl. shipping & gst)",
        "total sale amount (incl. shipping & gst)",
    ]
    agg_by_sum = [
        col
        for col in all_cols
        if (col not in agg_by_mean and col not in agg_by_first)
        and (col not in agg_by_cust and col != "sub order no")
    ]
    print("agg_lists created.")
    numeric_cols = agg_by_sum + agg_by_cust + agg_by_mean
    for col in numeric_cols:
        df_payments[col] = pd.to_numeric(df_payments[col])

    print(f"Numeric columns converted.")

    def mean_without_zero(series):
        np_array = series.to_numpy(dtype=float)
        cond = (np_array != 0) & ~np.isnan(np_array)
        if np.any(cond):
            return np_array[cond].mean()
        else:
            return 0

    agg_first = {k: "first" for k in agg_by_first}
    agg_mean = {k: "mean" for k in agg_by_mean}
    agg_sum = {k: "sum" for k in agg_by_sum}
    agg_custom = {k: mean_without_zero for k in agg_by_cust}
    agg_func = {**agg_mean, **agg_first, **agg_sum, **agg_custom}
    print("Aggregation dicts created")
    final_agg_dict = {
        k: v for k, v in agg_func.items() if k in all_cols
    }  ### We need to add a try, except block here to catch erors or missing fields

    grouped_payments = df_payments.groupby("sub order no", as_index=False).agg(
        final_agg_dict
    )
    print(f"payments grouped.")
    ## Merge orders and payments and create the final file
    mapped_orders = grouped_orders.merge(
        grouped_payments,
        on="sub order no",
        how="left",
        indicator=True,
        suffixes=["", "_payments"],
    )
    mapped_orders[numeric_cols] = mapped_orders[numeric_cols].fillna(0)
    print(f"orders and payments merged.")

    rename_dict = {
        "total sale amount (incl. shipping & gst)": "settled_sales",
        "total sale return amount (incl. shipping & gst)": "refunds",
        "supplier discounted price (incl gst and commision)": "sales_from_order_file",
        "supplier sku": "sku_id",
    }

    mapped_orders.rename(columns=rename_dict, inplace=True)

    ## Add logics for net_settled_sales, and net sales irrespective of settlements
    mapped_orders["net_settled_sales"] = (
        mapped_orders["settled_sales"] + mapped_orders["refunds"]
    )
    mapped_orders["net_sales_incl_unsettled"] = 0.0
    cond = mapped_orders["_merge"] == "both"
    mapped_orders["net_sales_incl_unsettled"] = np.where(
        cond,
        mapped_orders["net_settled_sales"],
        mapped_orders["sales_from_order_file"] * mapped_orders["quantity"],
    )
    
    mapped_orders['net_sales_incl_unsettled'] = np.where(
        mapped_orders['_merge'] == 'both',
        mapped_orders['net_settled_sales'],
        mapped_orders['sales_from_order_file']
    )

    print(f"mapped_orders created.")

    ## We need to add marketing spends at a daily level, so we need to create a per unit marketing spend dict, which we can then map with the main dataframe.
    ## Total marketing spend will then be per_unit marketing spend X quantity. However, if there are days when there is no sale but there was marketing spend,
    ## such days will not appear in the orders file. Hence, we need to add a dummy rows with missing dates, where we can map the marketing spend. This section
    ## creates the dummy rows
    marketing_dict = {k: -v for k, v in marketing_dict.items()}
    min_date = mapped_orders["order date"].min()
    max_date = mapped_orders["order date"].max()
    date_range = pd.date_range(start=min_date, end=max_date).date
    mapped_order_dates = set(
        pd.to_datetime(mapped_orders["order date"], errors="coerce").dt.date
    )
    missing_dates = [d for d in date_range if d not in mapped_order_dates]
    missing_dates_marketing_cost_df = pd.DataFrame(
        columns={
            "order date": missing_dates,
            "sub order no": ["zero orders"] * len(missing_dates),
            "quantity": [0] * len(missing_dates),
            "net_settled_sales": [0] * len(missing_dates),
            "net_sales_incl_unsettled": [0] * len(missing_dates),
        }
    )
    mapped_orders = pd.concat(
        [mapped_orders, missing_dates_marketing_cost_df], axis=0, ignore_index=True
    )
    for col in [
        "total sale return amount (incl. shipping & gst)",
        "total sale amount (incl. shipping & gst)",
    ]:
        numeric_cols.remove(col)

    numeric_cols.extend(["settled_sales", "refunds"])
    mapped_orders[numeric_cols] = mapped_orders[numeric_cols].fillna(0)
    mapped_orders[["settled_sales", "refunds"]] = mapped_orders[
        ["settled_sales", "refunds"]
    ].fillna(0)
    print(f"Marketing_dict created.")

    ## Now our mapped_orders dataframe has data against all dates. We can safely map it with our marketing files. The logic we use is that if quantity for any,
    ## day is 0, then take total marketing spend for that day and divide it by the number of rows, else, divide by total quantity for day. Since we are grouping
    ## by oreder date, this logic should be correct
    daily_row_count_dict = mapped_orders.groupby("order date")["order date"].count()
    daily_quantity_sold_dict = mapped_orders.groupby("order date")["quantity"].sum()

    marketing_per_unit_per_day_dict = (
        mapped_orders.groupby("order date")
        .agg({"quantity": "sum"})
        .assign(
            total_marketing_cost_for_day=lambda df: pd.Series(
                df.index.map(marketing_dict), index=df.index
            ).fillna(0),
            quantity_for_day=lambda df: df.index.map(daily_quantity_sold_dict),
            count_of_rows=lambda df: df.index.map(daily_row_count_dict),
            marketing_cost_per_day_per_unit=lambda df: df.apply(
                lambda row: row["total_marketing_cost_for_day"]
                * (
                    1 / row["quantity_for_day"]
                    if row["quantity_for_day"] != 0
                    else 1 / row["count_of_rows"]
                ),
                axis=1,
            ),
        )["marketing_cost_per_day_per_unit"]
        .to_dict()
    )
    print(f"marketing_per_unit_per_day_dict created.")

    cm1_cols = ["cogs", "net_settled_sales"]

    transaction_cols = [
        "fixed fee (incl. gst)",
        "return premium (incl gst)",
        "return premium (incl gst) of return",
        "meesho commission percentage",
        "meesho commission (incl. gst)",
        "meesho gold platform fee (incl. gst)",
        "meesho mall platform fee (incl. gst)",
        "fixed fee (incl. gst).1",
        "return shipping charge (incl. gst)",
        "gst compensation (prp shipping)",
        "shipping charge (incl. gst)",
        "waivers (excl. gst)",
        "net other support service charges (excl. gst)",
    ]

    overheads_cols = [col for col in agg_by_sum if col not in transaction_cols and col != 'final settlement amount']
    codb_cols = transaction_cols + overheads_cols

    mapped_orders["cogs"] = mapped_orders["sku_id"].map(cogs_dict).fillna(0)
    mapped_orders["cm1"] = mapped_orders[cm1_cols].sum(axis=1)
    mapped_orders["codb_unit_cost"] = mapped_orders[transaction_cols].sum(axis=1)
    mapped_orders["codb_overheads"] = mapped_orders[overheads_cols].sum(axis=1)
    mapped_orders["cm2"] = (
        mapped_orders["cm1"]
        + mapped_orders["codb_unit_cost"]
        + mapped_orders["codb_overheads"]
    )
    mapped_orders = mapped_orders.assign(
        marketing_per_unit=lambda df: df["order date"].map(
            marketing_per_unit_per_day_dict
        ),
        count_of_rows=lambda df: df["order date"].map(daily_row_count_dict),
        marketing=lambda df: df.apply(
            lambda row: row["marketing_per_unit"]
            * (row["quantity"] if row["quantity"] != 0 else row["count_of_rows"]),
            axis=1,
        ),
    )
    print(f"Orders mapped.")
    mapped_orders["cm3"] = mapped_orders["cm2"] + mapped_orders["marketing"]

    return mapped_orders


##### ------- NOTES ------
##### Structure of cogs_dict for the future
# 	cogs_dict = {
# 	'sku_1': {
# 					'date1': 'cogs1',
# 					'date2': 'cogs2',
# 					....
# 					},
# 	'sku_2': {
# 					'date': 'cogs',
# 					.....
# 					}
# 	}


##### A similar structure can be expected for marketing_dict in the future
