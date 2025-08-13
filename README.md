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
4. Inputs:
		a. order_df: A DataFrame prepared using the clean_orders function described earlier. Orders could be from order file or the mtr file
        b. payments_df: Consolidated payments DataFrame
        c. return_df: A DataFrame with FBA returns
        d. replacement_df: A Dataframe with FBA replacements
        e. daily_marketing_df: A Dataframe with daily marketing spends. Note, this file will have only two columns - date and spend
        e. cogs_df: A dataframe containing monthwise COGS for each sku
        f. start_date: The date from which to start the reconciliation
        g. end_date: The date until which to do the reconciliation
        h. mtr_file_bool: a Boolean input. Use False for FBA all orders file (default value) and True for MTR files
        i. mkt: Boolean flag to determine whether to use the marketing amount from payments or separate marketing file will be provided. False = use marketing cost from payments file
        j. incl_tax: Boolen flag to calculate this pre-tax or post tax. Default=True - include taxes
