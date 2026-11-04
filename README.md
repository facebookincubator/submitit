[![CI](https://github.com/facebookincubator/submitit/actions/workflows/ci.yaml/badge.svg)](https://github.com/facebookincubator/submitit/actions/workflows/ci.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pypi](https://img.shields.io/pypi/v/submitit)](https://pypi.org/project/submitit/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/submitit)](https://anaconda.org/conda-forge/submitit)
# submitit

submitit submits Python functions to [Slurm](https://slurm.schedmd.com/quickstart.html) from Python, and returns their results, errors and logs.
The same code runs locally.

```bash
pip install submitit  # or: conda install -c conda-forge submitit
```

```python
import submitit


def add(a, b):
    return a + b


executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10, slurm_partition="<your_partition>")
job = executor.submit(add, 5, 7)
print(job.job_id)
assert job.result() == 12
```

**Documentation: https://facebookincubator.github.io/submitit/**

## Contributors

By chronological order: Jérémy Rapin, Louis Martin, Lowik Chanussot, Lucas Hosseini, Fabio Petroni, Francisco Massa, Guillaume Wenzek, Thibaut Lavril, Vinayak Tantia, Andrea Vedaldi, Max Nickel, Quentin Duval, Rushil Patel. [Contributions](.github/CONTRIBUTING.md) are welcome.

## License

Submitit is released under the [MIT License](LICENSE).
