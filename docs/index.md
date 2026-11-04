# submitit

submitit submits Python functions to [Slurm](https://slurm.schedmd.com/quickstart.html) from Python, and returns their results, errors and logs.
The same code runs locally.

## Install

```bash
pip install submitit  # or: conda install -c conda-forge submitit
```

Install it in the environment your jobs should use: jobs run with the Python interpreter of the process that submits them.
Requires Python 3.11+.

## Submit a job

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

- `AutoExecutor` submits to Slurm when `srun` is available, and otherwise runs the job as a subprocess of the current machine (`executor.cluster` tells which).
- `folder` receives the pickled function, its result and the logs, with `%j` replaced by the job id. On Slurm, it must be on a filesystem shared with the compute nodes (not `/tmp`), otherwise jobs fail without logs.
- `job.result()` waits for the job and returns its output. It raises a `FailedJobError` with the traceback if the function raised, or its parent `UncompletedJobError` with the end of the logs if the job died without result (out of memory, cancelled, node failure).
- `executor.submit` raises a `FailedJobError` with the sbatch error if Slurm rejects the job.

## Set resources

```python
executor.update_parameters(
    timeout_min=60,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=1,
    mem_gb=32,
    slurm_partition="<your_partition>",  # sinfo -s lists them
    slurm_account="<your_account>",
)
```

- Always set `timeout_min`: the default is 5 minutes on Slurm and 2 locally.
- Unprefixed parameters work with every executor, see [`AutoExecutor`](#submitit.AutoExecutor).
- `slurm_` parameters are the [`SlurmExecutor`](#submitit.SlurmExecutor) parameters with a prefix (`partition` becomes `slurm_partition`), ignored when running locally. `slurm_additional_parameters` passes options submitit does not support.
- Executor arguments (`slurm_python`, `slurm_max_num_timeout`, `slurm_max_pickle_size_gb`) go to `submitit.AutoExecutor(folder, ...)`, not to `update_parameters`.

## Submit many jobs

To submit many jobs as one Slurm job array, use `executor.map_array` or `with executor.batch():`, see [Job arrays](examples.md#job-arrays).

## Follow a job

- `job.state` (e.g. `PENDING`, `RUNNING`), `job.done()` and `job.cancel()` follow and stop the job.
- `job.stdout()` and `job.stderr()` return its logs (`None` until it starts), written to `job.paths.stdout` and `job.paths.stderr`, e.g. `logs/1234/1234_0_log.out`.
- On Slurm, jobs keep running after your script exits, see the [FAQ](faq.md#how-do-i-retrieve-or-cancel-a-job-from-another-process) to recover them.

## Debug

- `submitit.AutoExecutor(folder="logs/%j", cluster="debug")` runs each job in the current process when its result is requested: breakpoints work, and a failure opens a post-mortem debugger.
- `cluster="local"` runs jobs as subprocesses of the current machine, even on a Slurm cluster.
- `job.exception()` returns the error of a failed job instead of raising it.
- On Slurm, job states are polled up to every 10 minutes, so a killed job can take that long to be detected: `job.done(force_check=True)` checks now.
- The [FAQ](faq.md) lists common errors and their fixes.

## Scope

- Each submission is a cluster job, with its own logs, requeued after preemption and [checkpointed](checkpointing.md) if it supports it.
- submitit is not a command-line tool (use [Hydra's submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher/) for that), nor a workflow engine: it does not manage dependencies between jobs (pass `slurm_dependency` yourself).
- Schedulers other than Slurm can be supported through [plugins](plugins.md).

```{toctree}
:hidden:

examples
checkpointing
structure
faq
nevergrad
plugins
api
```
