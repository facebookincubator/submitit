#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import typing as tp
from pathlib import Path

from setuptools import find_packages, setup

# Requirements

requirements = {}
for name in ["main", "dev"]:
    requirements[name] = Path(f"requirements/{name}.txt").read_text().splitlines()


## Version

init_str = Path("submitit/__init__.py").read_text()
match = re.search(r"^__version__ = \"(?P<version>[\w\.]+?)\"$", init_str, re.MULTILINE)
assert match is not None, "Could not find version in submitit/__init__.py"
version = match.group("version")


## Description


def _replace_relative_links(regex: tp.Match[str]) -> str:
    """Converts relative links into links to master
    so that links on Pypi long description are correct
    """
    string: str = regex.group()
    link = regex.group("link")
    name = regex.group("name")
    if not link.startswith("http") and Path(link).exists():
        githuburl = f"github.com/facebookincubator/submitit/blob/{version}"
        string = f"[{name}](https://{githuburl}/{link})"
    return string


pattern = re.compile(r"\[(?P<name>.+?)\]\((?P<link>\S+?)\)")
long_description = Path("README.md").read_text(encoding="utf-8")
long_description = re.sub(pattern, _replace_relative_links, long_description)


## Setup

setup(
    name="submitit",
    version=version,
    description="Python 3.6+ toolbox for submitting jobs to Slurm",
    author="Facebook AI Research",
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/facebookincubator/submitit",
        "Tracker": "https://github.com/facebookincubator/submitit/issues",
    },
    packages=find_packages(),
    install_requires=requirements["main"],
    extras_require={"dev": requirements["dev"]},
    # Mark the package as compatible with types.
    # https://mypy.readthedocs.io/en/latest/installed_packages.html#making-pep-561-compatible-packages
    package_data={"submitit": ["py.typed"]},
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: System :: Distributed Computing",
        "Development Status :: 5 - Production/Stable",
    ],
)
