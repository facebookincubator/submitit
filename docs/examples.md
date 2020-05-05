# Examples

## Explained example - Initial "add" exemple with a few more comments:
```python
import submitit

def add(a, b):
    return a + b

# the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
# The specified folder is used to dump job information, logs and result when finished
# %j is replaced by the job id at runtime
log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder=log_folder)
# The AutoExecutor provides a simple abstraction over SLURM to simplify switching between local and slurm jobs (or other clusters if plugins are available).
# specify sbatch parameters (here it will timeout after 4min, and run on dev)
# This is where you would specify `gpus_per_node=1` for instance
# Cluster specific options must be appended by the cluster name:
# Eg.: slurm partition can be specified using `slurm_partition` argument. It
# will be ignored on other clusters:
executor.update_parameters(timeout_min=4, slurm_partition="dev")
# The submission interface is identical to concurrent.futures.Executor
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for the submitted function to complete and returns its output
# if ever the job failed, job.result() will raise an error with the corresponding trace
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
```

## Job arrays

`submitit` supports the submission of [Slurm job arrays](https://slurm.schedmd.com/job_array.html) through the `executor.map_array` method.

If you want to submit many jobs at once, this is the **preferred way to go** because:
 - it can submit all jobs in only 1 call to slurm (avoids flooding it).
 - it is faster than submitting all jobs independently.
 - it lets you define a cap on how many jobs can run in parallel at any given time, so you can send thousands of jobs without breaking the scheduler, as long as you leave a reasonable value for this parallelism.

Here is an example on how to submit 4 additions at once, with at most 2 jobs running in parallel at any given time:
```python
a = [1, 2, 3, 4],
b = [10, 20, 30, 40]
executor = submitit.AutoExecutor(folder=log_folder)
# the following line tells the scheduler to only run\
# at most 2 jobs at once. By default, this is several hundreds
executor.update_parameters(slurm_array_parallelism=2)
jobs = executor.map_array(add, a, b)  # just a list of jobs
```

In comparison to standard jobs, job arrays have IDs like formatted as `<array job id>_<array task id>` (Eg: `17390420_15`) where the job id is
common to all the submitted jobs, and the task id goes from 0 to the `N - 1` where `N` is the number of submitted jobs.

**Note**:  `map_array` has no equivalent in `concurent.futures` (`map` is similar but has a different return type)

**Warning**: when running `map_array`, `submitit` will create one pickle per job.
If you have big object in your functions (like a full pytorch model) you should serialize it once
and only pass its path to the submitted function.

### Job arrays through a context manager

If you submit multiple jobs through a `for` loop like this one:
```python
jobs = []
for arg in whatever:
    job = executor.submit(myfunc, arg)
    jobs.append(job)
```
You can easily update it to batch the jobs into one array with exactly one extra line, by adding a batch context manager:
```python
jobs = []
with executor.batch():
    for arg in whatever:
        job = executor.submit(myfunc, arg)
        jobs.append(job)
```
This way, adding the `with` context to any existing code will convert it to an array submission,
the submission being triggered when leaving the context.

This allows to submit job arrays when the functions need many arguments and keywords arguments.

**Disclaimers**:
- within the context, you won't be allowed to interact with the jobs methods and attributes (nor even print it)! This is because the jobs are only submitted when leaving the context: inside the context, the jobs are like empty shells. You can however store the jobs in a list for instance, and access their attributes and methods after leaving the batch context.
- within the context, you can't update the executor parameters either (since all jobs must be submitted with the same settings)
- any error within the context will just cancel the whole submission.
- this option is still experimental and may undergo some changes in the future.


## Concurrent jobs

You can submit several jobs in parallel, and check their completion with the `done` method:
```python
import submitit
import time

executor = submitit.AutoExecutor(folder="log_test")
executor.update_parameters(timeout_min=1, slurm_partition="dev")
jobs = [executor.submit(time.sleep, k) for k in range(1, 11)]

# wait and check how many have finished
time.sleep(5)
num_finished = sum(job.done() for job in jobs)
print(num_finished)  # probably around 2 have finished, given the overhead

# then you may want to wait until all jobs are completed:
outputs = [job.result() for job in jobs]
```

Notice that this is straightforward to convert to multi-threading:
```python
import time
from concurrent import futures
with futures.ThreadPoolExecutor(max_workers=10) as executor:  # This is the only real difference
    jobs = [executor.submit(time.sleep, k) for k in range(1, 11)]
    time.sleep(5)
    print(sum(job.done() for job in jobs))  # around 4 or 5 should be over
    [job.result() for job in jobs]
    assert sum(job.done() for job in jobs) == 10  # all done
```

## Errors

Errors are caught and their stacktrace is recorded. When calling `job.result()`, a `FailedJobError` is raised with the available information:
```python
import submitit
from operator import truediv

executor = submitit.AutoExecutor(folder="log_test")
executor.update_parameters(timeout_min=1, slurm_partition="dev")
job = executor.submit(truediv, 1, 0)

job.result()  # will raise a FailedJobError stating the ZeroDivisionError with its stacktrace
full_stderr = job.stderr()  # recover the full stack trace if need be
# the stderr log is written in file job.get_logs_path("stderr")
```


## Working with commands

You should preferably submit pure Python function whenever you can. This would probably save you a lot of hassle.
Still, this is not always feasible. The class `submitit.helpers.CommandFunction` can help you in this case. It runs a
command in a subprocess and returns its stdout. It's main benefit is to be able to deal with errors and provide explicit errors.
(Note: `CommandFunction` runs locally, so you still need to submit it with an `Executor`
if you want to run it on slurm, see "Understanding the environment" below).
Note however that, because we use `subprocess` with `shell=False` under the hood, the command must be provided as a list and not a string.


By default, the function hides stdout and returns it at the end:
```python
import submitit
function = submitit.helpers.CommandFunction(["which", "python"])  # commands must be provided as a list!
print(function())  # This returns your python path (which you be inside your virtualenv)
# for me: /private/home/jrapin/.conda/envs/dfconda/bin/python
```

Some useful parameters of the `CommandFunction` class:
- `cwd`: to choose from which directory the command is run.
- `env`: to provide specific environment variables.
- `verbose`: set to `False` if you do not want any logging.

As an experimental feature, you can also provide arguments when calling the instance:
```python
print(submitit.helpers.CommandFunction(["which"])("pip"))  # will run  "which pip"
```


**Understanding the environment** - Make sure you have everything you need installed in your conda environment. Indeed, for its computation, Slurm uses
the active conda environment to submit your job:
```python
import submitit
function = submitit.helpers.CommandFunction(["which", "python"])
executor = submitit.AutoExecutor(folder="log_test")
executor.update_parameters(timeout_min=1, slurm_partition="dev")
job = executor.submit(function)

# The returned python path is the one used in slurm.
# It should be the same as when running out of slurm!
# This means that everything that is installed in your
# conda environment should work just as well in the cluster
print(job.result())
```


## Multi-tasks jobs

`submitit` support multi-tasks jobs (on one or several nodes).
You just need to use the `tasks_per_node` and `nodes` parameters.

```python
import submitit
from operator import add
executor = submitit.AutoExecutor(folder="log_test")
# 3 * 2 = 6 tasks
executor.update_parameters(tasks_per_node=3, nodes=2, timeout_min=1, slurm_partition="dev")
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.result())  # return [12, 12, 12, 12, 12, 12]
```

The same method will be executed in each task.
The typical usage is to use the task rank inside your submitted Callable to chunk the inputs, and attribute one chunk to each task.

We provide a `JobEnvironment` class, that gives access to this information (in a cluster-agnostic way).
```python
import submitit
from math import ceil

def my_func(inputs):
    job_env = submitit.JobEnvironment()
    print(f"There are {job_env.num_tasks} in this job")
    print(f"I'm the task #{job_env.local_rank} on the node {job_env.node}")
    print(f"I'm the task #{job_env.global_rank} in the job")
    num_items_per_task = int(ceil(len(inputs) / job_env.num_tasks))
    r = job_env.local_rank
    task_chunk = inputs[r * num_items_per_task: (r + 1) * num_items_per_task]
    return process(task_chunk)  # process only this chunk.
```

You can use the `task` method of a `Job` instance to access task specific information. A task is also a Job, so the Job's methods are available.

```python
import submitit

from operator import add
executor = submitit.AutoExecutor(folder="log_test")
# 3 * 2 = 6 tasks
executor.update_parameters(tasks_per_node=3, nodes=2, timeout_min=1, slurm_partition="dev")
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.task(2).result())  # Wait for task #2 result
print(job.task(2).stdout())  # Show task # stdout
print(job.result())  # Wait for all tasks and returns a list of results.
print(job.stdout())  # Concatenated stdout of all tasks
```


## Even more examples

TODO: share more examples, eg grid search over CIFAR-10
