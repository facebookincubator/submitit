# Examples

Each example runs locally when Slurm is not available.

## Job arrays

`map_array` submits one job per set of arguments as a single [Slurm job array](https://slurm.schedmd.com/job_array.html): one call to the scheduler, with a cap on the number of jobs running at the same time (`slurm_array_parallelism`).

```python
from operator import add

import submitit

executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10, slurm_array_parallelism=2)  # at most 2 jobs at a time
jobs = executor.map_array(add, [1, 2, 3, 4], [10, 20, 30, 40])
assert [job.result() for job in jobs] == [11, 22, 33, 44]
```

- On Slurm, job ids have the form `<array id>_<index>`, e.g. `17390420_3`.
- Each job pickles its own arguments: pass large objects (e.g. a model) as a file path.

`with executor.batch():` submits all the `executor.submit` calls of the block as one job array when the block exits, and unlike `map_array` it accepts keyword arguments:

```python
def power(x: int, exponent: int = 2) -> int:
    return x**exponent


jobs = []
with executor.batch():
    for exponent in range(4):
        jobs.append(executor.submit(power, 2, exponent=exponent))
assert [job.result() for job in jobs] == [1, 2, 4, 8]
```

Inside the block, using a job raises an `AttributeError` and `update_parameters` a `RuntimeError`, and an exception drops all the submissions of the block.

## Wait for jobs

`job.done()` tells whether a job has finished, and `submitit.helpers.as_completed` yields jobs as they finish:

```python
import time

import submitit

executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10)
jobs = [executor.submit(time.sleep, k) for k in range(3)]
print(sum(job.done() for job in jobs), "finished jobs")
for job in submitit.helpers.as_completed(jobs, poll_frequency=1):
    print(job.job_id, "finished")
```

Jobs provide `done()`, `result()` and `exception()` like `concurrent.futures.Future`.

## Wait with asyncio

`job.awaitable()` provides `wait()`, `result()` and `results()` as coroutines:

```python
import asyncio

import submitit


async def main() -> None:
    executor = submitit.AutoExecutor(folder="logs/%j")
    executor.update_parameters(timeout_min=10)
    jobs = [executor.submit(pow, 2, k) for k in range(4)]
    for result in asyncio.as_completed([job.awaitable().result() for job in jobs]):
        print(await result)


asyncio.run(main())
```

## Inspect a failed job

```python
from operator import truediv

import submitit

executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10)
job = executor.submit(truediv, 1, 0)
error = job.exception()  # waits for the job
assert isinstance(error, submitit.FailedJobError)
print(error)  # ZeroDivisionError traceback
print(error.__cause__)  # original ZeroDivisionError, when it can be pickled
print(job.stderr())  # full error log
```

`job.result()` still raises `FailedJobError`; its `__cause__` exposes the original
exception, so callers can inspect its type. If the exception cannot be pickled or
reconstructed, or the job used an older Submitit version, `__cause__` is `None` and
the formatted remote traceback remains available. Traceback objects are not serialized.

## Run a command

`submitit.helpers.CommandFunction` turns a command into a function returning its stdout, and raising a `FailedJobError` with its stderr if it fails:

```python
import os

import submitit

function = submitit.helpers.CommandFunction(
    ["python", "--version"],  # a list, not a string
    env={**os.environ, "MY_SETTING": "value"},
    verbose=False,
)
executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10)
job = executor.submit(function)
print(job.result())
```

- `verbose=True` (default) also prints the command and its output.
- `env` replaces the whole environment: merge it with `os.environ` to keep `PATH`.
- `cwd` sets the directory the command runs from.
- Arguments of the call are appended to the command, keyword arguments as `--key=value`.

## Multi-task jobs

With `tasks_per_node` (and `nodes`), every task runs the function, and `submitit.JobEnvironment()` gives each task its rank:

```python
import submitit


def square_chunk(inputs: list[int]) -> list[int]:
    env = submitit.JobEnvironment()
    return [x**2 for x in inputs[env.global_rank :: env.num_tasks]]  # one chunk per task


executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=10, nodes=1, tasks_per_node=2)
job = executor.submit(square_chunk, list(range(10)))
assert job.results() == [[0, 4, 16, 36, 64], [1, 9, 25, 49, 81]]  # one result per task
print(job.task(1).stdout())  # logs of task 1
```

`job.result()` raises an `AssertionError` on multi-task jobs (after waiting for them), use `job.results()`, and `job.stdout()` concatenates the logs of all tasks.

## PyTorch distributed

Request one task per GPU (`tasks_per_node=gpus_per_node`), and in the function call `submitit.helpers.TorchDistributedEnvironment().export()` before `torch.distributed.init_process_group(backend="nccl")`, which uses the default `env://` initialization, see the [full example](https://github.com/facebookincubator/submitit/blob/main/docs/examples/torch_distributed.py).
