# How it works

1. `executor.submit(function, *args, **kwargs)` pickles the function and its arguments in the job folder, and submits a script running them with `sbatch` (or starts a subprocess, locally).
2. The job runs `python -u -m submitit.core._submit <folder>` from the submission directory: it unpickles the function, calls it, and pickles its output or its error.
3. `job.result()` loads this output, or raises the error.

Files of a job, in its folder (`%j`: job id, `%t`: task id):

| File | `job.paths` attribute | Content |
|---|---|---|
| `%j_submitted.pkl` | `submitted_pickle` | function and arguments |
| `%j_submission.sh` | `submission_file` | sbatch script (Slurm) |
| `%j_%t_log.out` | `stdout` | stdout of each task |
| `%j_%t_log.err` | `stderr` | stderr of each task |
| `%j_%t_result.pkl` | `result_pickle` | output or error of each task |

submitit never deletes these files: clean the folder when you no longer need them.
