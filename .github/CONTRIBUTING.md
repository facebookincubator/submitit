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
2. Create a virtual environment and activate it: `make venv && . venv/bin/activate`
3. If you've added code please add tests.
4. If you've changed APIs, please update the documentation.
5. Ensure the test suite passes: `make test`
6. Make sure your code lints: `make pre_commit`. You can run this automatically on commit by making it a hook: `make register_pre_commit`
7. When ready you can run the full test suits ran on CI with `make -k integration`
8. If you haven't already, complete the Contributor License Agreement ("CLA").

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
We use black coding style with a generous 110 line length.

## License
By contributing to _submitit_, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
