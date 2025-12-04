# Contributing to _submitthem_
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process
There is no proof _submitit_ is still actively used by FAIR researcher and engineers.
We can’t garanty that _submitthem_ will be actively used by any researcher and engineers.
All bugs tracking and feature plannings are public.
_submitthem_ will NOT be updated to keep up with Slurm/PBS versions and to fix bug,
We don't have any major feature planned ahead.

## Pull Requests
We actively welcome pull requests and will review them as quickly as possible, subject to our availability.

1. Fork the repo and create your branch from `main`.
2. Create a virtual environment and activate it: `make venv && . venv/bin/activate`
3. If you've added code please add tests.
4. If you've changed APIs, please update the documentation.
5. Ensure the test suite passes: `make test`
6. Make sure your code lints: `make pre_commit`. You can run this automatically on commit by making it a hook: `make register_pre_commit`
7. When ready you can run the full test suits ran on CI with `make -k integration`

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Coding Style
We use black coding style with a generous 110 line length.

## License
By contributing to _submitthem_, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
