# submitit

submitit submits Python functions to [Slurm](https://slurm.schedmd.com/quickstart.html) from Python, and returns their results, errors and logs.
The same code runs locally.

## Install

```bash
pip install submitit  # or: conda install -c conda-forge submitit
pip install git+https://github.com/facebookincubator/submitit@main  # development version
```

Install it in the environment your jobs should use: jobs run with the Python interpreter of the process that submits them, so on Slurm this environment must be on a filesystem shared with the compute nodes.
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
- `job.result()` waits for the job and returns its output. It raises a `submitit.core.utils.FailedJobError` with the traceback if the function raised or the job reached its time limit, or its parent `UncompletedJobError` with the end of the logs if the job died without result (out of memory, cancelled, node failure).
- `executor.submit` raises a `FailedJobError` with the sbatch error if Slurm rejects the job.
- The job imports your code from the submission directory, not from the directory of your script, see the [FAQ](faq.md#my-job-fails-with-modulenotfounderror-or-importerror) if imports fail.

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
- Unprefixed parameters are accepted by every executor, see [`AutoExecutor`](#submitit.AutoExecutor). The local executor only uses `timeout_min`, `tasks_per_node`, `gpus_per_node` and `stderr_to_stdout`.
- `slurm_` parameters are the [`SlurmExecutor`](#submitit.SlurmExecutor) parameters with a prefix (`partition` becomes `slurm_partition`), ignored when running locally. `slurm_setup` runs shell commands before the job (e.g. `module load`), and `slurm_additional_parameters` passes options submitit does not support.
- Executor arguments (`slurm_python`, `slurm_max_num_timeout`, `slurm_max_pickle_size_gb`) go to `submitit.AutoExecutor(folder, ...)`, not to `update_parameters`.

## Submit many jobs

To submit many jobs as one Slurm job array, use `executor.map_array` or `with executor.batch():`, see [Job arrays](examples.md#job-arrays).

## Follow a job

- `job.state` (e.g. `PENDING`, `RUNNING`), `job.done()` and `job.cancel()` follow and stop the job.
- `job.stdout()` and `job.stderr()` return its logs (`None` until it starts), written to `job.paths.stdout` and `job.paths.stderr`, e.g. `logs/1234/1234_0_log.out`.
- On Slurm, jobs keep running after your script exits, see the [FAQ](faq.md#how-do-i-retrieve-or-cancel-a-job-from-another-process) to recover them.

## Debug

- `submitit.AutoExecutor(folder="logs/%j", cluster="debug")` runs each job in the current process the first time `result()`, `done()`, `wait()` or `exception()` is called: breakpoints work, a failure opens a post-mortem debugger, and `exception()` returns the original exception.
- `cluster="local"` runs jobs as subprocesses of the current machine, even on a Slurm cluster.
- `job.exception()` returns the error of a failed job instead of raising it, see [Inspect a failed job](examples.md#inspect-a-failed-job), and `job.paths.submission_file` is the sbatch script, see [How it works](structure.md).
- On Slurm, a killed job can take up to 10 minutes to be detected, see [`Job.done`](#submitit.Job.done).
- The [FAQ](faq.md) lists common errors and their fixes.

## Scope

- Any picklable callable can be submitted, including lambdas and functions defined in the submitting script.
- Each submission is a cluster job, with its own logs, requeued after preemption (if the cluster signals it) and [checkpointed](checkpointing.md) if it supports it.
- submitit is not a command-line tool (use [Hydra's submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher/) for that), nor a task queue: it does not schedule jobs itself, nor manage dependencies between them (pass e.g. `slurm_dependency=f"afterok:{job.job_id}"` yourself).
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
