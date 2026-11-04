# Contributing to _submitit_
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process
_submitit_ is actively used by FAIR researcher and engineers.
All bugs tracking and feature plannings are public.
_submitit_ will be updated to keep up with Slurm versions and to fix bug,
but we don't have any major feature planned ahead.


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Create a virtual environment, activate it, then install the package and the pre-commit hooks: `pip install -e .[dev] && pre-commit install`
3. If you've added code please add tests.
4. If you've changed APIs, please update the documentation.
5. Ensure the test suite passes: `pytest submitit`
6. Make sure your code lints: `mypy submitit && pylint submitit`
7. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
We format with ruff through pre-commit (`pre-commit run --all-files`), with a generous 110 line length.

## Releasing
After bumping `__version__` in `submitit/__init__.py` on `main`:
```bash
VERSION=$(python -c "import submitit; print(submitit.__version__)")
git tag $VERSION
python submitit/test_documentation.py  # makes README links absolute for PyPI
SOURCE_DATE_EPOCH=$(git log -n1 --format=%cd --date=unix) flit publish --setup-py
git checkout HEAD -- README.md
git push origin $VERSION
```

## License
By contributing to _submitit_, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
