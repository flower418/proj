from setuptools import find_namespace_packages, setup


setup(
    name="pseudospectrum_tracker",
    version="0.1.0",
    description="Neural-enhanced pseudospectrum contour tracking",
    packages=find_namespace_packages(include=["src*"]),
    install_requires=[],
)
