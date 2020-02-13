#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=7.0', ]

setup(
    author="Trevor James Manz",
    author_email='trevor.j.manz@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python package for converting images to zarr chunked arrays.",
    entry_points={
        'console_scripts': [
            'img2zarr=img2zarr.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='img2zarr',
    name='img2zarr',
    packages=find_packages(include=['img2zarr', 'img2zarr.*']),
    test_suite='tests',
    url='https://github.com/hubmapconsortium/img2zarr',
    version='0.0.0',
    zip_safe=False,
)
