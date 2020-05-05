# Structure

## Under the hood

When you submit a function and its arguments, it will return a Job instance to you. Under the hood, the function and arguments are
pickled, and the submission functions run a batch file which will load the pickled object, compute the function
with the provided arguments, and pickle the output of the function into a new file. Whenever this file becomes available, your Job instance
will be able to recover it. The computation in the cluster will use the current conda environment, so make sure everything you need is installed (including `submitit`).

If the computation failed and we are able to catch the error, then the trace will be dumped and available to you through the Job instance as well.
However, if it could not be catched, you will be notified as well, but will probably need to look into the logs to understand what happened.

For each job, you will therefore usually end up with a task file `<job_id>_submitted.pkl`, an output file `<job_id>_result.pkl`,
a batch file `batchfile_<uuid>.sh`, a stdout log file `<job_id>_<task_id>_log.out` and a stderr log file `<job_id>_<task_id>_log.err`, where the uuid
is created by `submitit`, and the id is the job id from slurm. The Job instance helps you link all of this together (see `job.job_id`).

## Main objects

Here are some information about the main objects defined in `submitit`, but you can always refer to the docstrings if you need details.

### Executor

The executor is your interface for submitting jobs. Its role is to save/dump the job you want to submit,
then, in Slurm case for instance, to create the sbatch file for running submitting the job.
Its main methods are:
 - `submit(function, *args, **kwargs)`: submits a function for run on the cluster with given parameters, and returns
   a `Job` instance.
 - `map_array(function, *iterables)`: submits a function several times for run on the cluster with different parameters
   (pulled from the iterables), and returns a list of `Job` instances. On Slurm, this uses [job arrays](https://slurm.schedmd.com/job_array.html),
   which are the preferred options for submitting large number of jobs in parallel, since they are better handled by the scheduler.
   The `slurm_array_parallelism` parameter of `executor.update_parameters` controls how many jobs will be able to run in parallel on Slurm cluster.
 - `update_parameters(**kwargs)`: sets or updates parameters for the cluster. Only a subset is implemented but
   it can be easily improved with more parameters. We have homogenized some parameter names, to use the same
   parameters for slurm and other clusters (eg, use `gpus_per_node=2`, that historically corresponds to `--gres=gpu:2` for slurm, but is now `--gpus-per-node` as well).
   If you misspell a name, the function will raise an exception with all allowed parameters (this can be useful if you are looking for
   an argument ;) )

`submitit` has a plugin system so that several executor implementations can be provided. There are currently several implementations:
- `AutoExecutor` which **we advise to always use** for submititting to clusters. This executor chooses the best available plugin to use depending on your environment. The aim is to be able to use the same code an several clusters.
- `SlurmExecutor` which only works for slurm, and should be used through `AutoExecutor`.
- `LocalExecutor` which provides a way to test job submission locally through multiprocessing.
- `DebugExecutor` which mocks job submission and does all the computation in the same process.


### Job

Jobs are processes running on the cluster. The `Job` class implements methods for checking the state and the results, as well as
raised exceptions within the job. This class tends to replicate the main element of the  `concurrent.Future` API and adds some more.
Its main methods and attributes are:
 - `job_id`: ID of the job in slurm (`str`).
 - `state`: the current state of the job (Eg: `RUNNING`, `PENDING`).
 - `done`: whether your job is finished.
 - `result()`: waits for completion and returns the result of the computation, or raises an exception if it failed.
 - `cancel()`: cancels the job
 - `<stdout,stderr>()`: returns the text contained in stdout/stderr logs.
 - `submission()`: returns the content of your submission (see `DelayedSubmission` object below)

### Job environment

`submitit.JobEnvironment` is a handle to access information relevant to the current job such as its id. It therefore has the following attributes:
`job_id`, `num_tasks`, `num_nodes`, `node`, `global_rank`, `local_rank`.

### helpers

This module implements convenient functions/classes for use with `submitit`:
 - `CommandFunction`: a class transforming a shell command into a function, so as to be able to submit it as well (see examples below).
 - `Checkpointable`: base class implementing a very basic example of checkpointing (`checkpoint` method). More on this on the Checkpointing section.
 - `FunctionSequence`: A function that computes sequentially the output of other functions. This can be used
 to compute several independent results sequentially on a unique job, and it implements checkpointing for free.


### DelayedSubmission

This is the class which contains all information about the job. You will only have to deal with it if you do
custom checkpointing (see below). Its main attributes are:
- `function`: the function (or callable) to be called
- `args`: the positional arguments
- `kwargs`: the keyword arguments
It is basically used exactly as the `submit` method of an `executor`: `DelayedSubmission(func, arg1, arg2, kwarg1=12)`
