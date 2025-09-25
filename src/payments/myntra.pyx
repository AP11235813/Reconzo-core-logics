import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict

def read_data(
	_path: str,
	filename_dict: dict={
						"GST Sales": "sales",
						"GST RT": "returns",
						"RTO": "rto",
						"PG Forward": "payment_forward",
						"PG Reverse": "payment_reverse",
						"combined_file": "combined_file"
						}
	) -> dict:
	"""
	This function reads all relevant data as a dictionary.

	Parameters:
	-----------
	_path: string
		Full path of the folder where the data resides

	filename_dict: dictionary
		A dictionary containing names of the files and standard names as the key-value pairs

	Output:
	-------
	A dictionary of the form {std_filename: DataFrame, ...}

	"""

	data_dict = defaultdict(pd.DataFrame)
	folder_path=Path(_path)

	for file in folder_path.iterdir():
		print(file)
		_matches_ = [w for w in filename_dict.keys() if w in file.name]
		print(f"Matches found: {_matches_}")
		best_match = max(_matches_, key=len) if _matches_ else None
		print(f"best match found: {best_match}")
		if best_match:
			file_name = filename_dict.get(best_match)
			df = pd.read_csv(file)
			if file_name in data_dict:
				temp_df = data_dict[file_name]
				df = pd.concat([df, temp_df], axis=0)

			data_dict[file_name] = df

	return dict(data_dict)

def get_return_and_rto_dates(
	data_dict: dict
	) -> dict:
	"""
	Complies the return and RTO dates into one column for ease.
	"""

	file_types = ["rto"]
	for file_type in file_types:
		df = data_dict[file_type]
		df["status"] = "canceled"
		print(df.columns)
		mask = df["order_rto_date"].isna()
		df.loc[~mask, "status"] = "rto"
		df["order_rto_date"] = df["order_rto_date"].combine_first(df["order_cancel_date"])
		data_dict[file_type] = df

	return data_dict

def get_settlement_date(
	data_dict: dict,
	file_types: list=["payment_reverse", "payment_forward"],
	cols_to_combine: dict={
							"settlement_date": ["settlement_date_prepaid_payment", "settlement_date_postpaid_payment"]
							}
	) -> dict:
	"""
	combines settlement date into one single column
	"""

	for file_type in file_types:
		df = data_dict[file_type]
		for col in list(cols_to_combine.keys()):
			combine_1 = cols_to_combine.get(col)[0]
			combine_2 = cols_to_combine.get(col)[1]
			df[col] = df[combine_1].combine_first(df[combine_2])

		data_dict[file_type] = df

	return data_dict

def get_required_cols(
	data_dict: dict,
	col_name_dict: dict={
							"sales": ["order_id", "order_created_date", "sku_id", "article_type"],
							"returns": ["shipment_id", "order_created_date", "fr_refunded_date", "Qty", "sku_id", "article_type"],
							"rto": ["order_id", "order_created_date", "order_rto_date", "quantity", "status"],
							"payment_forward": ["order_release_id", "settlement_date", "mrp", "total_discount_amount", "seller_product_amount", "igst_amount", "cgst_amount", "sgst_amount","tcs_amount", "tds_amount", "platform_fees", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee", "total_tax_on_logistics", "shipment_zone_classification", "total_actual_settlement"],
							"payment_reverse": ["order_release_id", "settlement_date", "mrp", "total_discount_amount", "seller_product_amount", "igst_amount", "cgst_amount", "sgst_amount","tcs_amount", "tds_amount", "platform_fees", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee", "total_tax_on_logistics", "shipment_zone_classification", "total_actual_settlement"],
							"combined_file": ["order_release_id", "Order_Type"]
						}
	) -> dict:
	"""
	Gets relevant columns from the data within the data dictionary
	
	Parameters:
	-----------
	data_dict: Dictionary
		data_dict created from read_data

	order_col_names: list
		List of required columns from the order file

	rto_col_names: list
		List of required columns from the rto file

	return_col_names: list
		List of required columns from the return file

	forward_pmt_col_names: list
		List of required columns from the forward payment file

	reverse_pmt_col_names: list
		List of required columns from the reverse payment file

	"""

	for file_type, df in data_dict.items():
		req_cols = col_name_dict.get(file_type)
		df = df[req_cols]
		if file_type == "returns":
			df["status"] = "returned"

		data_dict[file_type] = df

	return data_dict

def add_quantity_in_sales(
	data_dict: dict
	) -> dict:
	"""
	Adds a quantity column in the sales file
	"""
	sales=data_dict["sales"]
	sales["quantity"] = 1
	data_dict["sales"] = sales

	return data_dict

def calculate_platform_discount(
	data_dict: dict
	) -> dict:
	"""
	calcuates platform funded discounts
	"""

	file_types = ["payment_reverse", "payment_forward"]
	for file_type in file_types:
		df=data_dict[file_type]
		df["platform_funded_discount"] = df["mrp"] - df["total_discount_amount"] - df["seller_product_amount"]
		data_dict[file_type] = df

	return data_dict

def assign_signs_for_deductions(
	data_dict: dict,
	deduction_cols = ["total_discount_amount", "tcs_amount", "tds_amount", "platform_fees", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee", "total_tax_on_logistics"]
	) -> dict:
	"""
	Assigns a sign for each payment deduction
	"""

	file_types=["payment_reverse", "payment_forward"]
	for file_type in file_types:
		df = data_dict[file_type]
		df[deduction_cols] = df[deduction_cols] * (-1)
		data_dict[file_type] = df

	return data_dict

def reverse_entries_for_payment_reverse(
	data_dict: dict
	) -> dict:
	"""
	Reverses entries for payment_reverse.
	"""

	file_type = "payment_reverse"
	non_reverse_entries = ["order_release_id", "shipping_fee", "total_tax_on_logistics"]
	df = data_dict[file_type]
	cols = [col for col in df.columns if col not in non_reverse_entries]
	df[cols] = df[cols] * (-1)
	data_dict[file_type] = df

	return data_dict

def correct_dates(
	data_dict: dict,
	date_col_dict: dict={
							"sales": ["order_created_date"],
							"returns": ["order_created_date", "fr_refunded_date"],
							"rto": ["order_created_date", "order_rto_date"],
							"payment_forward": ["settlement_date"],
							"payment_reverse": ["settlement_date"]
						}
	) -> dict:
	"""
	Corrects dates in valrious columns for various files and gets them in the right format
	"""

	compact_date_files = ["sales", "returns", "rto"]

	for file_type in date_col_dict.keys():
		df=data_dict.get(file_type)
		date_cols = date_col_dict.get(file_type)
		for col in date_cols:
			if file_type in compact_date_files:
				df[col] = pd.to_datetime(df[col], errors='coerce', format="%Y%m%d")

			df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

		data_dict[file_type] = df

	return data_dict

def add_tax_on_commissions(
	data_dict: dict
	) -> dict:
	"""
	Calculates tax on commissions. Myntra provides this inclusive of taxes
	"""

	file_types = ["payment_reverse", "payment_forward"]
	for file_type in file_types:
		df=data_dict[file_type]
		df["platform_fees"]=df["platform_fees"] / 1.18
		df["tax_on_commissions"]=df["platform_fees"] * 0.18
		data_dict[file_type] = df

	return data_dict

def standardize_names(
	data_dict: dict,
	rename_dict: dict={
						"sales": ["order_id", "order_date", "sku", "sub-category", "quantity"],
						"returns": ["order_id", "order_date", "return/rto/cancelled_date", "returned_quantity", "sku", "sub-category", "order_status"],
						"rto": ["order_id", "order_date", "return/rto/cancelled_date", "returned_quantity", "order_status"],
						"payment_forward": ["order_id", "settlement_date", "mrp", "discount", "net_realized_sale", "igst", "cgst", "sgst","tcs", "tds", "commission", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee", "tax_on_logistics", "zone", "actual_settlement", "tax_on_commissions"],
						"payment_reverse": ["order_id", "settlement_date", "mrp", "discount", "net_realized_sale", "igst", "cgst", "sgst","tcs", "tds", "commission", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee", "tax_on_logistics", "zone", "actual_settlement", "tax_on_commissions"],
						"combined_file": ["order_release_id", "Order_Type"]
	}
	) -> dict:
	"""
	standardizes names of columns for all fee types
	"""

	for file_type, df in data_dict.items():
		col_names = rename_dict.get(file_type)
		df.columns = col_names
		data_dict[file_type] = df

	return data_dict

def get_order_status(
	data_dict: dict
	) -> dict:
	"""
	Gets the order status of an order by merging with returns and rto files
	"""

	sales_df = data_dict["sales"]
	return_df = data_dict["returns"]
	rto_df = data_dict["rto"]

	rto_dict = rto_df.set_index("order_id")["order_status"].to_dict()
	return_dict = return_df.set_index("order_id")["order_status"].to_dict()

	sales_df["order_status"] = sales_df["order_id"].map(return_dict)
	mask = sales_df["order_status"].isna()
	sales_df.loc[mask, "order_status"] = sales_df.loc[mask, "order_status"].map(rto_dict)
	mask = sales_df["order_status"].isna()
	sales_df.loc[mask, "order_status"] = "delivered"
	data_dict["sales"] = sales_df

	return data_dict

def create_single_payment_file(
	data_dict: dict
	) -> dict:
	
	"""
	merge forward and reverse dataframes into 1 single file
	"""

	pg_forward = data_dict["payment_forward"]
	pg_reverse = data_dict["payment_reverse"]
	non_sum_cols = ["settlement_date", "order_id"]
	agg_func = {k: ("last" if k in non_sum_cols else "sum") for k in pg_forward.columns}
	final_payment = (
		pd.concat([pg_forward, pg_reverse], axis=0)
		.groupby("order_id", as_index=False)
		.agg(agg_func)
		)
	data_dict["final_payment"] = final_payment

	return data_dict

def map_payments(
	data_dict: dict
	) -> dict:
	"""
	Map payments to the order_file
	"""

	payments = data_dict["final_payment"]
	sales = data_dict["sales"]
	sales = sales.merge(payments, on="order_id", how="left", indicator=True)
	sales["payment_status"] = sales["_merge"].replace(
														{
															"both": "settled", 
															"left_only": "unsettled"
														}
													)

	data_dict["mapped_orders"] = sales

	return data_dict

def get_returned_qty(
	data_dict: dict
	) -> dict:
	"""
	Add returned quantity to the mapped payment file
	"""
	mapped_orders = data_dict["mapped_orders"]
	returns = data_dict["returns"]
	return_qty_dict = returns.set_index("order_id")["returned_quantity"].to_dict()
	mapped_orders["returned_quantity"] = mapped_orders["order_id"].map(return_qty_dict)
	data_dict["mapped_orders"] = mapped_orders

	return data_dict

def calculate_net_delivered_quantity(
	data_dict: dict
	) -> dict:
	"""
	Calculates net delivered quantity
	"""

	mapped_orders = data_dict["mapped_orders"]
	mapped_orders["net_delivered_quantity"] = mapped_orders["quantity"] - mapped_orders["returned_quantity"]
	data_dict["mapped_orders"] = mapped_orders

	return data_dict

def calculate_final_settlement(
	data_dict: dict
	) -> dict:
	"""
	Calculates the total deduction for each order
	"""
	mapped_orders=data_dict["mapped_orders"]

	cols_to_sum = ["net_realized_sale", "tcs", "tds", "commission", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee", "tax_on_logistics", "tax_on_commissions"]
	mapped_orders["final_settlement"] = mapped_orders[cols_to_sum].sum(axis=1)
	data_dict["mapped_orders"] = mapped_orders

	return data_dict

def create_fee_estimator(
	data_dict: dict
	) -> dict:
	"""
	Creates a fee estimator - based on business context
	"""

	mapped_orders = data_dict["mapped_orders"]
	settled = mapped_orders[mapped_orders["payment_status"] == "settled"]
	fee_estimate_dict = defaultdict(dict)

	fee_types = ["commission", "shipping_fee", "fixed_fee", "pick_and_pack_fee", "payment_gateway_fee"]
	settled["commission_pct"] = settled["commission"] / settled["net_realized_sale"]
	settled["pg_pct"] = settled["payment_gateway_fee"] / settled["net_realized_sale"]

	fee_estimate_dict["commission"] = (
		settled
		.groupby("sku")["commission_pct"]
		.mean()
		.to_dict()
		)
	fee_estimate_dict["payment_gateway_fee"] = {"unsettled": settled["pg_pct"].mean()}
	for fee in fee_types:
		if fee in ["commission", "payment_gateway_fee"]:
			continue
		else:
			fee_estimate_dict[fee] = {"unsettled": settled[fee].mean()}

	return dict(fee_estimate_dict)

def estimate_unsettled_orders(
	data_dict: dict,
	fee_estimate_dict: dict,
	complete_data_dict: dict
	) -> dict:
	"""
	Calculates fees for unsettled orders
	"""
	mapped_orders = data_dict["mapped_orders"]
	unsettled = mapped_orders[mapped_orders["payment_status"] == "unsettled"].copy()
	settled = mapped_orders[mapped_orders["payment_status"] == "settled"].copy()
	all_sales = complete_data_dict["sales"].copy()
	print(all_sales)

	unsettled_oid = list(set(unsettled["order_id"]))
	mask = all_sales["order_id"].isin(unsettled_oid)
	all_sales = all_sales[mask].copy()

	req_cols = ["order_created_date", "sku_id", "mrp", "article_type", "discount", "seller_price", "igst_amt", "cgst_amt", "sgst_amt", "tcs_amount", "tds_amount"]
	all_sales = all_sales[req_cols].copy()
	all_sales.rename(
		columns={
		"order_created_date": "order_date",
		"sku_id": "sku",
		"article_type": "sub-category",
		"seller_price": "net_realized_sale",
		"igst_amount": "igst",
		"cgst_amount": "cgst",
		"sgst_amount": "sgst",
		"tcs_amount": "tcs",
		"tds_amount": "tds"
		},
		inplace=True
		)
	all_sales['quantity'] = 1
	all_sales['order_status'] = "delivered"
	all_sales["payment_status"] = "unsettled"

	for fee in fee_estimate_dict:
		if fee == "commission":
			all_sales['commission_pct'] = all_sales["sku"].map(fee_estimate_dict.get(fee))
			all_sales[fee] = all_sales["commission_pct"] * all_sales["net_realized_sale"]
			all_sales.drop(columns=["commission_pct"], inplace=True)
		elif fee == "payment_gateway_fee":
			all_sales["pg_pct"] = all_sales["payment_status"].map(fee_estimate_dict.get(fee))
			all_sales[fee] = all_sales["net_realized_sale"] * all_sales["pg_pct"]
			all_sales.drop(columns=["pg_pct"], inplace=True)
		else:
			all_sales[fee] = all_sales["payment_status"].map(fee_estimate_dict.get(fee))

	all_sales["tax_on_commissions"] = all_sales["commission"] * 0.18
	logistics_cols = ["shipping_fee", "pick_and_pack_fee", "fixed_fee", "payment_gateway_fee"]
	all_sales["tax_on_logistics"] = all_sales[logistics_cols].sum(axis=1) * 0.18

	data_dict["unsettled"] = all_sales
	data_dict["settled"] = settled

	return data_dict

def create_mis(
	data_dict: dict
	) -> dict:
	"""
	Creates an MIS based on estimated values
	"""
	settled = data_dict["settled"]
	unsettled = data_dict["unsettled"]
	mis = pd.concat([settled, unsettled], axis=0)
	data_dict["mis"] = mis

	return data_dict

def create_payment_recon_rangita(
	data_dict: dict,
	complete_data_dict: dict
	) -> dict:
	"""
	Create payment recon in the format required by Rangita
	"""
	mapped_orders = data_dict["mapped_orders"]
	returns = data_dict["returns"]
	rto = data_dict["rto"]
	combined = data_dict["combined_file"]
	sales = complete_data_dict["sales"]

	sale_dict = sales.groupby("order_id")["seller_price"].first().to_dict()
	returns["tag"] = 1
	rto["tag"] = 1
	mask = combined["Order_Type"] == "NOD"
	combined = combined[mask].copy()
	combined["Order_Type"] = "SPF"

	return_dict = returns.set_index("order_id")["tag"].to_dict()
	rto_dict = rto.set_index("order_id")["tag"].to_dict()
	spf_dict = combined.groupby("order_release_id")["Order_Type"].first()

	df = mapped_orders.copy()
	df["Sale"] = df["order_id"].map(sale_dict)
	df["return_tag"] = df["order_id"].map(return_dict)
	df["rto_tag"] = df["order_id"].map(rto_dict)
	df["spf_tag"] = df["order_id"].map(spf_dict)

	df["RT"] = df["Sale"].where(~df["return_tag"].isna(), 0)
	df["RTO"] = df["Sale"].where(~df["rto_tag"].isna(), 0)

	df["Net Sale"] = df["Sale"] - df["RT"] - df["RTO"]
	df["Commission"] = (df["commission"] + df["tax_on_commissions"]) * (-1)
	df["Logistics"] = (df["shipping_fee"] + df["pick_and_pack_fee"] + df["fixed_fee"] + df["payment_gateway_fee"] + df["tax_on_logistics"]) * (-1)
	df["TCS"] = df["tcs"] * (-1)
	df["TDS"] = df["tds"] * (-1)
	df["Settled Amount"] = df["actual_settlement"]
	df["SPF"] = df["Settled Amount"].where(df["spf_tag"] == "SPF", 0)
	df["Total"] = df["Settled Amount"] + df["Commission"] + df["Logistics"] + df["TCS"] + df["TDS"]
	cols = ["Net Sale", "Total", "SPF"]
	df[cols] = df[cols].fillna(0)
	df["Net Receivable"] = df["Net Sale"] - df["Total"] - df["SPF"]

	conditions = [
					df["Net Receivable"].round(0) == 0,
					df["Net Receivable"].round(0) > 0,
					df["Net Receivable"].round(0) < 0
					]
	values = ["Fully Paid", "Pending Payment", "Partially Settled"]
	df["Remarks"] = np.select(conditions, values)
	d_today = datetime.today()
	df["Ageing(Days)"] = (d_today - df["order_date"]).dt.days

	aging = [
				df["Ageing(Days)"] < 30,
				df["Ageing(Days)"].between(30, 60),
				df['Ageing(Days)'] > 60
				]
	buckets = ["<30", "30-60", ">60"]
	df["Slab"] = np.select(aging, buckets)


	df.rename(columns={"order_date": "Order Date", "order_id": "Order No."}, inplace=True)
	df.drop(columns=["return_tag", "rto_tag", "spf_tag"], inplace=True)

	req_cols = ["Order Date", "Order No.", "Sale", "RT", "RTO", "Net Sale", "Commission", "Logistics", "TCS", "TDS", "Settled Amount", "Total", "SPF", "Net Receivable", "Remarks", "Ageing(Days)", "Slab"]

	df = df[req_cols]
	data_dict["payment_recon_rangita"] = df

	return data_dict

def map_myntra_payments(
	_path_: str
	) -> Tuple[Dict, Dict]:
	"""
	Main function to control the output
	"""

	print(f"\n** IMPORTANT **\n---------------\n")
	print(f"Files need to contain certain keywords for python to understand the file type. See below:")
	print(f"|----------------------------------------------------")
	print(f"|File type (All required)   |   Keyword to contain  |")
	print(f"|----------------------------------------------------")
	print(f"|Forward payments           |   PG Forward          |")
	print(f"|----------------------------------------------------")
	print(f"|Reverse payments           |   PG Reverse          |")
	print(f"|----------------------------------------------------")
	print(f"|GST Sale report            |  	GST Sales           |")
	print(f"|----------------------------------------------------")
	print(f"|GST Returns                |   GST RT              |")
	print(f"|----------------------------------------------------")
	print(f"|RTO files                  |   RTO                 |")
	print(f"|----------------------------------------------------")
	print(f"|A combined payment file    |   combined_file       |")
	print(f"|----------------------------------------------------")

	time.sleep(10)

	complete_data_dict=read_data(_path_)
	data_dict = complete_data_dict.copy()
	data_dict=get_return_and_rto_dates(data_dict)
	data_dict=get_settlement_date(data_dict)
	data_dict=get_required_cols(data_dict)
	data_dict=add_quantity_in_sales(data_dict)
	data_dict=assign_signs_for_deductions(data_dict)
	data_dict=reverse_entries_for_payment_reverse(data_dict)
	data_dict=correct_dates(data_dict)
	data_dict=add_tax_on_commissions(data_dict)
	data_dict=standardize_names(data_dict)
	data_dict=get_order_status(data_dict)

	sales = data_dict["sales"]
	print(f"No. of rows in sales: {sales.index.size}")

	data_dict=create_single_payment_file(data_dict)
	data_dict=map_payments(data_dict)
	data_dict=calculate_final_settlement(data_dict)
	data_dict=get_returned_qty(data_dict)
	data_dict=calculate_net_delivered_quantity(data_dict)
	data_dict=calculate_final_settlement(data_dict)
	fee_estimate_dict=create_fee_estimator(data_dict)
	data_dict=estimate_unsettled_orders(data_dict, fee_estimate_dict, complete_data_dict)
	data_dict=create_mis(data_dict)
	data_dict=create_payment_recon_rangita(data_dict, complete_data_dict)

	some_path = Path(_path_)
	file_name_mid = some_path.stem

	files_to_save = ["mapped_orders", "mis", "payment_recon_rangita"]

	for file in files_to_save:
		print(f"Saving files: {file}")
		x = data_dict.get(file)
		store_loc = some_path.parent.name + "-" + file_name_mid + "-" + file + ".csv"
		x.to_csv(store_loc, index=False)

	return data_dict, fee_estimate_dict



















