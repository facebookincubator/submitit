# FAQ

## My job fails with `ModuleNotFoundError` or `ImportError`

The job does not run your script: it runs `python -m submitit.core._submit` from your submission directory, so imports relying on the directory of your script, or on a `sys.path.append` in it, fail.
Solutions, from the most robust:

- install your code as a package in the environment (`pip install -e .`);
- add its directory to `PYTHONPATH` before running the submitting script;
- submit from the directory containing your modules;
- as a last resort, call `sys.path.append` inside the submitted function, before importing the module.

## My job runs code I edited after submitting it

Functions imported from a module are pickled by reference: the job imports the module as it is when the job starts, or is requeued.
Submit inside `with submitit.helpers.RsyncSnapshot(snapshot_dir):` to run jobs from a snapshot of your git repository (this does not cover installed packages).

## A module fails at import in the job, but not in my script

The job imports the modules of the submitted function when unpickling it, not in the order of your script: imports with side effects (e.g. a module that must be imported before another) may break.
Import them inside the submitted function, or in a module imported by it.

## Pickling fails

Some objects cannot be pickled (locks, open files, database connections, objects wrapping C resources), including as default arguments of the submitted function.
Create them inside the submitted function rather than passing them as arguments.
Prefer functions imported from a module: only their name is pickled, and tracebacks are clearer.

## My job is killed after a few minutes

Set `timeout_min`, see [Set resources](index.md#set-resources).

## My `logging` output does not appear in the logs

Only the submitted function runs in the job: logging configured by your script is not configured in the job, and the root logger only shows warnings by default.
Configure logging at the beginning of the submitted function.

## sbatch rejects an option that I did not set

`executor.submit` raises a `FailedJobError` with the sbatch error.
submitit always writes `--job-name`, `--nodes`, `--time`, `--signal`, `--wckey` and the log options (`--output`, `--error`, `--open-mode`), and the other options only when set.
See `wckey` and `gpus_per_node` in [`SlurmExecutor`](#submitit.SlurmExecutor) for workarounds.

## How do I load modules or set environment variables before the job starts?

```python
executor.update_parameters(slurm_setup=["module load cuda/12.1", "export HF_HOME=/scratch/hf"])
```

## How do I run the job with another Python or in a container?

```python
executor = submitit.AutoExecutor(folder="logs/%j", slurm_python="apptainer exec image.sif python")
```

## Jobs submitted from within a job fail, e.g. with `Unable to satisfy cpu bind request`

Submit inside [`submitit.helpers.clean_env()`](#submitit.helpers.clean_env).

## Processes started by my function ignore `terminate()`, with `Bypassing signal SIGTERM` in the logs

submitit makes the job log and ignore `SIGTERM`, and processes forked by your function (`multiprocessing` with the `fork` start method, the default on Linux before Python 3.14) inherit this handler.
Stop them with `kill()`, let them exit by themselves (`close()` and `join()` a pool), or start them with the `spawn` method.

## How do I retrieve or cancel a job from another process?

Recreate it from its folder and id, e.g. [`submitit.SlurmJob(folder="logs/%j", job_id="1234")`](#submitit.SlurmJob) (`job_id="1234_3"` for a job array, `tasks=range(n)` for a multi-task job), to check its `state`, wait for its `result()` or `cancel()` it.
Local jobs cannot be followed from another process: their job id is the id of their controlling process, and `kill -INT <job_id>` stops them (`kill` alone does not).

## Can I use torchrun?

No: submitit already starts one process per task through Slurm, see [PyTorch distributed](examples.md#pytorch-distributed).

## How does submitit compare to dask.distributed?

[`dask.distributed`](https://distributed.dask.org) runs tasks on a pool of workers, while submitit runs each submission as a cluster job.
For many small tasks, use dask (with [dask-jobqueue](https://jobqueue.dask.org) on Slurm), or group them in [job arrays](examples.md#job-arrays): many independent jobs overload the scheduler.
