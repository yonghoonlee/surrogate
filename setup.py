#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2021 Yong Hoon Lee

import os
from setuptools import find_packages, setup

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, "surrogate")

about = {}
with open(os.path.join(src_dir, "__about__.py")) as f:
    exec(f.read(), about)
with open(os.path.join(base_dir, "README.md")) as f:
    long_description = f.read()

metadata = dict(
    name = about['__title__'],
    version = about['__version__'],
    description = about['__description__'],
    long_description = long_description,
    author = about['__author__'],
    url = about['__uri__'],
    packages = find_packages(),
    install_requires = [
        'matplotlib',
        'numpy',
        'scipy',
        'smt',
        'pyDOE2',
        'pyyaml',
    ],
    python_requires = '>=3.8',
)
setup(**metadata)
