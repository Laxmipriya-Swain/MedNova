from setuptools import setup,find_packages
with open("req.txt") as f:
    requir = f.read().splitlines()

setup(
    name="Medical_book",
    packages=find_packages(),
    install_requires = requir,

)