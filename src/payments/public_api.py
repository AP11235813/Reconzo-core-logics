from .amazon import map_amazon_payments
from .myntra import map_myntra_payments
from .shopify import map_shopify_payments

# try:
# 	from .core import map_payments
# except Exception:
# 	def map_payments():
# 		print(f"Module import failed!")