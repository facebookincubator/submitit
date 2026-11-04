[![CircleCI](https://circleci.com/gh/facebookincubator/submitit.svg?style=svg)](https://circleci.com/gh/facebookincubator/workflows/submitit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pypi](https://img.shields.io/pypi/v/submitit)](https://pypi.org/project/submitit/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/submitit)](https://anaconda.org/conda-forge/submitit)
# Submit it!

## What is submitit?

Submitit is a lightweight tool for submitting Python functions for computation within a Slurm cluster.
It basically wraps submission and provide access to results, logs and more.
[Slurm](https://slurm.schedmd.com/quickstart.html) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters.
Submitit allows to switch seamlessly between executing on Slurm or locally.

### An example is worth a thousand words: performing an addition

From inside an environment with `submitit` installed:

```python
import submitit

def add(a, b):
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")
# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=1, slurm_partition="dev")
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
```

The `Job` class also provides tools for reading the log files (`job.stdout()` and `job.stderr()`).

If what you want to run is a command, turn it into a Python function using `submitit.helpers.CommandFunction`, then submit it.
By default stdout is silenced in `CommandFunction`, but it can be unsilenced with `verbose=True`.

**Find more examples [here](docs/examples.md)!!!**

Submitit is a Python 3.8+ toolbox for submitting jobs to Slurm.
It aims at running python function from python code.


## Install

Quick install, in a virtualenv/conda environment where `pip` is installed (check `which pip`):
- stable release:
  ```
  pip install submitit
  ```
- stable release using __conda__:
  ```
  conda install -c conda-forge submitit
  ```
- main branch:
  ```
  pip install git+https://github.com/facebookincubator/submitit@main#egg=submitit
  ```

You can try running the [MNIST example](docs/mnist.py) to check that everything is working as expected (requires sklearn).


## Documentation

The full documentation is available at **https://facebookincubator.github.io/submitit/**, including:

- [Getting started](docs/getting_started.md): submitting, inspecting and debugging jobs.
- [Parameters](docs/parameters.md): resources, Slurm options and environment setup.
- [Examples](docs/examples.md): for a bunch of examples dealing with errors, concurrency, multi-tasking etc...
- [Structure and main objects](docs/structure.md): to get a better understanding of how `submitit` works, which files are created for each job, and the main objects you will interact with.
- [Checkpointing](docs/checkpointing.md): to understand how you can configure your job to get checkpointed when preempted and/or timed-out.
- [Tips and caveats](docs/tips.md): for a bunch of information that can be handy when working with `submitit`.
- [FAQ](docs/faq.md): common errors and how to solve them.
- [Hyperparameter search with nevergrad](docs/nevergrad.md): basic example of `nevergrad` usage and how it interfaces with `submitit`.
- [Goals, non-goals and comparison with dask.distributed](docs/index.md#goals).


## Contributors

By chronological order: Jérémy Rapin, Louis Martin, Lowik Chanussot, Lucas Hosseini, Fabio Petroni, Francisco Massa, Guillaume Wenzek, Thibaut Lavril, Vinayak Tantia, Andrea Vedaldi, Max Nickel, Quentin Duval, Rushil Patel (feel free to [contribute](.github/CONTRIBUTING.md) and add your name ;) )

## License

Submitit is released under the [MIT License](LICENSE).
