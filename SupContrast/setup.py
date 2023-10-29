import os

import setuptools

if not os.path.exists("__init__.py"):
    open("__init__.py", 'a').close()

setuptools.setup(
    name="SupContrast",
    package_dir={"SupContrast": "."},
    packages=["SupContrast"]
)
