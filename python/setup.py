import io
import os
import sys

from setuptools import setup, find_packages

if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="fedml",
    version="0.7.8",
    author="FedML Team",
    author_email="ch@fedml.ai",
    description="A research and production integrated edge-cloud library for "
    "federated/distributed machine learning at anywhere at any scale.",
    long_description=io.open(
        os.path.join("../README.md"), "r", encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FedML-AI/FedML",
    keywords=[
        "distributed machine learning",
        "federated learning",
        "natural language processing",
        "computer vision",
        "Internet of Things",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    install_requires=requirements,
    package_data={"": ["py.typed"]},
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "fedml=fedml.cli.cli:cli",
        ]
    },
)
