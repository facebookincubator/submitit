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

**Find more examples [here](examples.md)!!!**


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

You can try running the [MNIST example](https://github.com/facebookincubator/submitit/blob/main/docs/mnist.py) to check that everything is working as expected (requires sklearn).


## Goals

The aim of this Python3 package is to be able to launch jobs on Slurm painlessly from *inside Python*, using the same submission and job patterns than the standard library package `concurrent.futures`.

Here are a few benefits of using this lightweight package:

 - submit any function, even lambda and script-defined functions.
 - raises an error with stack trace if the job failed.
 - requeue preempted jobs (Slurm only)
 - swap between `submitit` executor and one of `concurrent.futures` executors in a line, so that it is easy to run your code either on slurm, or locally with multithreading for instance.
 - checkpoints stateful callables when preempted or timed-out and requeue from current state (advanced feature).
 - easy access to task local/global rank for multi-nodes/tasks jobs.
 - same code can work for different clusters thanks to a plugin system.

Submitit is used by FAIR researchers on the FAIR cluster.
The defaults are chosen to make their life easier, and might not be ideal for every cluster.

## Non-goals

- a commandline tool for running slurm jobs. Here, everything happens inside Python. To this end, you can however use [Hydra](https://hydra.cc/)'s [submitit plugin](https://hydra.cc/docs/next/plugins/submitit_launcher) (version >= 1.0.0).
- a task queue, this only implements the ability to launch tasks, but does not schedule them in any way.
- being used in Python2! This is a Python3.8+ only package :)


## Comparison with dask.distributed

[`dask`](https://distributed.dask.org/en/latest/) is a nice framework for distributed computing. `dask.distributed` provides the same `concurrent.futures` executor API as `submitit`:

```python
from distributed import Client
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(processes=1, cores=2, memory="2GB")
cluster.scale(2)  # this may take a few seconds to launch
executor = Client(cluster)
executor.submit(...)
```

The key difference with `submitit` is that `dask.distributed` distributes the jobs to a pool of workers (see the `cluster` variable above) while `submitit` jobs are directly jobs on the cluster. In that sense `submitit` is a lower level interface than `dask.distributed` and you get more direct control over your jobs, including individual `stdout` and `stderr`, and possibly checkpointing in case of preemption and timeout. On the other hand, you should avoid submitting multiple small tasks with `submitit`, which would create many independent jobs and possibly overload the cluster, while you can do it without any problem through `dask.distributed`.

```{toctree}
:hidden:

examples
structure
checkpointing
tips
nevergrad
plugins
```
