from setuptools import setup, find_packages

setup(
    name="simple-markdown-crawler",
    packages=find_packages(exclude=['markdown']),
    include_package_data=True,
)