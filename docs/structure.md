# Structure

## Under the hood

When you submit a function and its arguments, it will return a Job instance to you. Under the hood, the function and arguments are
pickled, and the submission functions run a batch file which will load the pickled object, compute the function
with the provided arguments, and pickle the output of the function into a new file. Whenever this file becomes available, your Job instance
will be able to recover it. The computation in the cluster will use the current conda environment, so make sure everything you need is installed (including `submitit`).

If the computation failed and we are able to catch the error, then the trace will be dumped and available to you through the Job instance as well.
However, if it could not be catched, you will be notified as well, but will probably need to look into the logs to understand what happened.

For each job, you will therefore usually end up with a task file `<job_id>_submitted.pkl`, an output file `<job_id>_result.pkl`,
a submission file `<job_id>_submission.sh`, a stdout log file `<job_id>_<task_id>_log.out` and a stderr log file `<job_id>_<task_id>_log.err`, where the id is the job id from slurm. The Job instance helps you link all of this together (see `job.job_id`).
