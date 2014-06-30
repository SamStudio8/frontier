#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

README = open('README.rst').read()
CHANGELOG = open('CHANGELOG.rst').read()

requirements = [
    "numpy"
]

test_requirements = [
    "tox",
    "pytest",
    "nose",
    "python-coveralls",
]

setuptools.setup(
    name="frontier",
    version="0.1.1",
    url="https://github.com/samstudio8/frontier",

    description="Provides interfaces for the reading, storage and retrieval of large machine learning data sets for use with scikit-learn",
    long_description=README + '\n\n' + CHANGELOG,

    author="Sam Nicholls",
    author_email="sam@samnicholls.net",

    maintainer="Sam Nicholls",
    maintainer_email="sam@samnicholls.net",

    packages=setuptools.find_packages(),
    include_package_data=True,

    install_requires=requirements,

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 2.7',
    ],

    test_suite='tests',
    tests_require=test_requirements
)
