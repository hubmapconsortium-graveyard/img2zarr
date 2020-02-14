#!/usr/bin/env python
import io
import os
import re
from setuptools import setup, find_packages


def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def get_install_requirements(path):
    content = read(path)
    return [
        requirement
        for requirement in content.split("\n")
        if requirement != "" and not requirement.startswith("#")
    ]


def version(path):
    """Obtain the package version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(
        r"""^__version__ = ['"]([^'"]*)['"]""", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


DESCRIPTION = "Python package for converting images to zarr chunked arrays."
LONG_DESCRIPTION = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
NAME = "img2zarr"
PACKAGES = find_packages(include=["img2zarr", "img2zarr.*"])
AUTHOR = "Heath Patterson / Trevor Manz"
AUTHOR_EMAIL = "trevor.j.manz@gmail.com"
URL = "https://img2zarr.readthedocs.io."
DOWNLOAD_URL = "https://github.com/hubmapconsortium/img2zarr"
LICENSE = "MIT license"
INSTALL_REQUIRES = get_install_requirements("requirements.txt")
PYTHON_REQUIRES = ">=3.5"
DEV_REQUIRES = get_install_requirements("requirements_dev.txt")
VERSION = "0.0.0"


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={"console_scripts": ["img2zarr=img2zarr.cli:main"]},
    zip_safe=False,
)
