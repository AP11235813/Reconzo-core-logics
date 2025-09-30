Steps to install

1. Create a Python virtual environment (3.9 or above)
2. Run "pip install git+https://github/.com/AP11235813/Reconzo-core-logics.git@vx.x.x"
3. In your Python code / module, import the package:
	Following modules are currently active:
	(a) payments - contains two functions (i) map_amazon_payments (ii) map_myntra_payments
	(b) fee_audits - contains one function (i) complete_fee_audit_amazon

	For details on function and usage type "help(<function_name>)" in python.


Sample Python Code
	from payments import map_amazon_payments
	df = map_amazon_payments(order_df, payments_df, return_df, replacement_df, daily_marketing_df, cogs_df, start_date(optional), end_date(optional), mtr_file_bool(optional), mkt(optional), incl_tax(optional)) see Input Arguments below
	
	or

	import payments
	df = payments.map_amazon_payments(order_df, payments_df, return_df, replacement_df, daily_marketing_df, cogs_df, start_date(optional), end_date(optional), mtr_file_bool(optional), mkt(optional), incl_tax(optional))

Module information:
1. The main function is amazon.map_payments which resides within the payments package
2. This function maps amazon orders with amazon payments. Also maps cogs (which is sent separately)
3. Output is a single dataframe.
4. payments_df is a consolidated payments file (structure remains the same, with the exception of settlement_date, which is "ffill" to remove blanks and add a date against each line)
5. return_df and replacement_df is read directly from Amazon - no changes
6. daily_marketing_df - should have only 2 columns - date and spend. dates should be unique.
7. cogs_df - derived from Reconzo's internal Additional Data sheet. Sheet is read as is, all processing happens within the code.

Inputs Arguments:
1. order_df: A DataFrame prepared using the clean_orders function described earlier. Orders could be from order file or the mtr file
2. payments_df: Consolidated payments DataFrame
3. return_df: A DataFrame with FBA returns
4. replacement_df: A Dataframe with FBA replacements
5. daily_marketing_df: A Dataframe with daily marketing spends. Note, this file will have only two columns - date and spend
6. cogs_df: A dataframe containing monthwise COGS for each sku
7. start_date: (Optional) The date from which to start the reconciliation
8. end_date: (Optional) The date until which to do the reconciliation
9. mtr_file_bool: (Optional, default: False) a Boolean input. Use False for FBA all orders file (default value) and True for MTR files
10. mkt: (Optional, default: False) Boolean flag to determine whether to use marketing amount from payments or separate file. False = use marketing cost from payments file
11. incl_tax: (Optional, default: True) Boolen flag to calculate this pre-tax or post tax. Default=True - include taxes
