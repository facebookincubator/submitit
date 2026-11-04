# Getting started

## Submit a function

Install submitit (see [Install](index.md#install)) in the environment your jobs should use: jobs run with the same Python interpreter as the process that submits them.

```python
import submitit


def add(a, b):
    return a + b


executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10, slurm_partition="dev")
job = executor.submit(add, 5, 7)
print(job.job_id)
assert job.result() == 12
```

- `AutoExecutor` submits to Slurm when `srun` is available, and otherwise runs the job in a subprocess on the current machine, so the same code works on a laptop and on the cluster.
- `folder` receives the pickled function, the result and the logs of each job; `%j` is replaced by the job id. On Slurm it must be on a filesystem shared with the compute nodes (not `/tmp`).
- `update_parameters` sets the resources of the next submissions, see [Parameters](parameters.md). Parameters prefixed by `slurm_` are ignored when not running on Slurm.
- `job.result()` waits for the job to finish and returns its output. If the function raised, it raises a `FailedJobError` containing the traceback from the job, see [Errors](examples.md#errors).

The job runs from the directory you submitted from, see [Under the hood](structure.md#under-the-hood) for the full mechanism.

## Inspect a job

```python
print(job.state)
print(job.paths.folder)  # submission, result and log files
print(job.paths.submission_file)  # generated sbatch script on Slurm
print(job.stdout())
print(job.stderr())
```

A job can be recovered from another process with its folder and id, e.g. `submitit.SlurmJob(folder="logs/%j", job_id="1234")`, to check its state, wait for its result or `cancel()` it.

## Submit many jobs

`map_array` submits one job per set of arguments, as a single [Slurm job array](https://slurm.schedmd.com/job_array.html):

```python
jobs = executor.map_array(add, [1, 2, 3], [10, 20, 30])
assert [job.result() for job in jobs] == [11, 22, 33]
```

Prefer it to a loop of `submit` calls, which floods the scheduler with independent submissions; `with executor.batch():` turns such a loop into a job array. See [Job arrays](examples.md#job-arrays).

## Run locally or in the current process

```python
local_executor = submitit.AutoExecutor(folder="logs/%j", cluster="local")
debug_executor = submitit.AutoExecutor(folder="logs/%j", cluster="debug")
assert debug_executor.submit(add, 1, 2).result() == 3
```

- `cluster="local"` runs jobs as subprocesses of the current machine, even on a Slurm cluster.
- `cluster="debug"` runs jobs in the current process when their result is requested: breakpoints work, and a failure opens a post-mortem debugger.

## Next steps

- [Parameters](parameters.md): resources, Slurm options and environment setup.
- [Examples](examples.md): errors, concurrency, multi-task jobs, commands.
- [Checkpointing](checkpointing.md): requeue jobs after preemption or timeout.
- [FAQ](faq.md): common errors and how to solve them.
