# FAQ

## My job fails with `ModuleNotFoundError` or `ImportError`

Functions defined in a module are pickled by reference: the job imports that module again before running the function.
The job runs from the directory you submitted from, but not your script, so imports that only work because of your script's location or a `sys.path.append` fail in the job.
Solutions, from the most robust:

- install your code as a package in the environment (`pip install -e .`);
- add its directory to `PYTHONPATH` before running the submitting script;
- submit from the directory containing your modules;
- as a last resort, call `sys.path.append` inside the submitted function, before importing the module.

See [Tips and caveats](tips.md) for other pickling pitfalls.

## Pickling fails, or hits a recursion error

Some objects cannot be pickled (locks, open files, database connections, objects wrapping C resources), including as default arguments of the submitted function.
Create them inside the submitted function rather than passing them as arguments, and prefer functions imported from a module to functions defined in the submitting script, which also give more explicit tracebacks.

## My job is killed after a few minutes

Jobs that do not set `timeout_min` get a short default time limit, see [Generic parameters](parameters.md#generic-parameters).

## My `logging` output does not appear in the logs

Only the submitted function runs in the job: logging configured by your script is not configured in the job, and the root logger only shows warnings by default.
Configure logging at the beginning of the submitted function.

## sbatch rejects an option that I did not set

submitit always writes `--job-name`, `--nodes`, `--time`, `--signal`, `--wckey` and the log options (`--output`, `--error`, `--open-mode`), and the other options only when set:

- `Invalid wckey specification`: set `slurm_wckey=""`;
- `Invalid generic resource (gres) specification` with `gpus_per_node`: your cluster may not support `--gpus-per-node`, request GPUs with e.g. `slurm_gres="gpu:2"` instead;
- any option submitit does not support: use `slurm_additional_parameters`, see [Slurm parameters](parameters.md#slurm-parameters).

## How do I load modules or set environment variables before the job starts?

Use `slurm_setup`, see [Slurm parameters](parameters.md#slurm-parameters).

## How do I run the job with another Python or in a container?

Use `slurm_python`, see [Executor arguments](parameters.md#executor-arguments).

## Jobs submitted from within a job fail, e.g. with `Unable to satisfy cpu bind request`

A job submitted from a Slurm job inherits its `SLURM_*` environment variables.
Submit inside `submitit.helpers.clean_env()`, which removes them and restores them on exit:

```python
with submitit.helpers.clean_env():
    job = executor.submit(evaluate, checkpoint)
```

## Processes started by my function ignore `terminate()`, with `Bypassing signal SIGTERM` in the logs

submitit makes the job log and ignore `SIGTERM`, and processes forked by your function (`multiprocessing` with the `fork` start method, the default on Linux before Python 3.14) inherit this handler.
Stop them with `kill()`, let them exit by themselves (`close()` and `join()` a pool), or start them with the `spawn` method.

## How do I retrieve or cancel a job from another process?

Recreate it from its folder and id, see [Inspect a job](getting_started.md#inspect-a-job).
For the local executor, the job id is the process id.

## Can I use torchrun?

No: submitit already starts one process per task through Slurm.
Request one task per GPU (`tasks_per_node=gpus_per_node`) and initialize PyTorch distributed with `submitit.helpers.TorchDistributedEnvironment`, see [PyTorch distributed initialization](examples.md#pytorch-distributed-initialization).

## Does submitit work with other schedulers than Slurm?

Only Slurm, local and debug executors are provided; other schedulers need a [plugin](plugins.md).
