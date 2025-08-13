Steps to install

1. Download the installable binaries for your OS from Releases
2. Create a Python virtual environment (3.9 or above)
3. Run "pip install <filename>"
4. In your Python code / module, import the package:
	from payments import amazon
	df = map_payments(args) see below
	
	or

	import payments
	df = amazon.map_payments(args)

Module information:
1. The main function is amazon.map_payments which resides within the payments package
2. This function maps amazon orders with amazon payments. Also maps cogs (which is sent separately)
3. Output is a single dataframe 

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
