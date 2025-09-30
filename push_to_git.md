Before pushing to GitHub ensure the following:
---------------------------------------------

1. Change version numbers in pyprojects.toml
2. Update version number in setup.py
3. Check for dependencies. If installing any custom packages - ensure that this is mentioned in the dependency

Then from python_projects do the following:

git add .
git commit -m "<some message here explaining the version difference>"
git tag -a vx.x.x -m "Release message"
git push origin main --tags

Version Naming convention:
--------------------------

vx1.x2.x3

Where:
------
x1: Number of products that have been released - Products are Payments, Fee_audit, Inventory, Return_recon, MIS, profitability etc.
x2: Major release - whenver a new platform is added, increment this by 1
x3: actual release number


