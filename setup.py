from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "src/payments/*.pyx",  # path to your Cython files
        compiler_directives={"language_level": "3"},
    ),
    packages=["payments"],
    package_dir={"": "src"},
)
