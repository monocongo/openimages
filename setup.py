import os
from setuptools import setup, find_packages

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f"{parent_dir}/README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="openimages",
    version="0.0.1",
    author="James Adams",
    author_email="monocongo@gmail.com",
    description="Tools for downloading computer vision datasets from Google's OpenImages dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monocongo/openimages",
    python_requires=">=3.6",
    provides=[
        "openimages",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "boto3",
        "cvdata",
        "lxml",
        "pandas",
        "requests",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "oi_download_dataset=openimages.download:_entrypoint_download_dataset",
            "oi_download_images=openimages.download:_entrypoint_download_images",
        ]
    },
)
