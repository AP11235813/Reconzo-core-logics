import pandas
from datetime import timedelta, date

def xtracover_inventory_recon(sales_df: pd.DataFrame, inventory_df: pd.DataFrame, shipped_units_df: pd.DataFrame, start_date: str = '2025-07-01') -> pd.DataFrame:
	sales_df.columns = sales_df.columns.astype(str).str.strip().str.lower()
	inventory_df.columns = inventory_df.columns.astype(str).str.strip().str.lower()
	inventory_df.drop_duplicates(subset=['imei'], keep='last', inplace=True)
	shipped_units_df.columns = shipped_units_df.columns.astype(str).str.lower().str.strip()

	shipped_units_df['imei1'] = shipped_units_df['imei1'].astype(str).str.split('.').str[0]
	req_cols = [
		'subpurchase_orderno', 
		'imei1', 
		'suborderconfirmeddatetime'
		]

	shipped_df = shipped_units_df[req_cols].copy()
	sales_df = sales_df.merge(shipped_df, on='subpurchase_orderno', how='left').fillna("")
	sales_df.rename(columns={
		'imei1': 'imei', 
		'suborderconfirmeddatetime_y': 'confirmation_date'
		}, inplace=True)

	req_cols = [
		'subpurchase_orderno', 
		'subquantity', 
		'imei', 
		'subpurchaseorderstatus', 
		'confirmation_date',
		'shippingstatename'
		]

	inventory_movement_df = sales_df[req_cols].copy()
	inventory_movement_df['confirmation_date'] = pd.to_datetime(inventory_movement_df['confirmation_date']).dt.date

	rename_dict = {
	'subpurchase_orderno': 'sub-order-id',
	'subquantity': 'gross_quantity',
	'subpurchaseorderstatus': 'status',
	'shippingstatename': 'state'
	}

	inventory_movement_df.rename(columns=rename_dict, inplace=True)
	inventory_movement_df['status'] = inventory_movement_df['status'].replace(r'SubOrder', "", regex=True)

	inventory_movement_df = inventory_movement_df.assign(
		flag=lambda x: x['status'].isin(['Delivered','Shipped']) * -1,
		shipped_quantity=lambda x: x['flag'] * x['gross_quantity']
		).drop(columns='flag')

	inventory_df['quantity'] = 1
	inventory_df['imei'] = inventory_df['imei'].astype(str)
	inventory_df = inventory_df.merge(inventory_movement_df, on='imei', how='left', suffixes=['', '_sales'], indicator=True)
	inventory_df[['gross_quantity', 'shipped_quantity']] = inventory_df[['gross_quantity', 'shipped_quantity']].fillna(0)
	inventory_df['final_quantity'] = inventory_df['quantity'] + inventory_df['shipped_quantity']

	inventory_df['productname'] = inventory_df['productname'].str.title()
	inventory_df['brand'] = inventory_df['productname'].str.split(' ').str[0]
	inventory_df['brand'] = inventory_df['brand'].str.split(':').str[0]
	inventory_df.loc[inventory_df['brand'] == 'Hp', 'brand'] = 'HP'
	inventory_df.loc[inventory_df['brand'] == 'Iphone', 'brand'] = 'Apple'

	inventory_df.rename(columns={
		'productname': 'product_name',
		'confirmation_date': 'shipped_date'
		}, inplace=True)
	start_date = pd.to_datetime(start_date).date()
	agg_func={
		'quantity': 'sum',
		'brand': 'first',
		'product_name': 'first'
	}
	rename_dict = {
		'shipped_quantity': 'sales',
	}
	starting_df = inventory_df.groupby('imei', as_index=False).agg(agg_func).fillna(0)
	print(f"{starting_df=}")
	starting_df.rename(columns={
		'quantity': 'closing balance'
		}, inplace=True)
	starting_df['date'] = start_date - timedelta(days=1)
	today = date.today()
	list_of_df = []
	date_col = start_date
	agg_func['shipped_quantity'] = 'sum'
	opening_balance_mapper = starting_df.set_index('imei')['closing balance'].to_dict()
	print(f"{opening_balance_mapper=}")
	blank_start = starting_df.copy()
	blank_start['opening balance'] = 0
	blank_start.drop(columns=['closing balance'], inplace=True)

	while date_col < today:
		df = blank_start.copy()
		cond = inventory_df['shipped_date'] == date_col
		df_temp = inventory_df[cond].groupby('imei', as_index=False).agg(agg_func)
		df = df.merge(df_temp, on='imei', how='left', suffixes=['', '_drop'])
		df.rename(columns=rename_dict, inplace=True)

		df['opening balance'] = df['imei'].map(opening_balance_mapper)
		other_cols = [
			'receipts', 
			'customer returns', 
			'vendor returns', 
			'iwt', 
			'lost', 
			'found', 
			'damaged', 
			'disposed', 
			'other events', 
			'unknown events'
			]
		
		for col in other_cols:
			df[col] = 0

		all_cols = other_cols + ['sales', 'opening balance']
		df['closing balance'] = df[all_cols].sum(axis=1)
		df['date'] = date_col
		opening_balance_mapper = df.set_index('imei')['closing balance'].to_dict()
		date_col += timedelta(days=1)
		print(f"{df=}")
		list_of_df.append(df)

	final_df = pd.concat(list_of_df, axis=0, ignore_index=True)

	return inventory_df, final_df
