from setuptools import find_packages, setup

setup(
    name="loom-db",
    version="0.0.0a",
    packages=find_packages(),
    install_requires=["numpy", "lru-dict"]
)
