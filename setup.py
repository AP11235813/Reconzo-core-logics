# from setuptools import setup
# from Cython.Build import cythonize

# setup(
#     ext_modules=cythonize(
#         "src/payments/*.pyx",  # path to your Cython files
#         compiler_directives={"language_level": "3"},
#     ),
#     packages=["payments"],
#     package_dir={"": "src"},
# )


from setuptools import setup, find_packages
from Cython.Build import cythonize
import glob

ext_modules = cythonize(
    glob.glob("src/payments/*.pyx"),
    compiler_directives={"language_level": "3"}
)

setup(
    name="payments",
    version="0.2.4",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    include_package_data=True,
)