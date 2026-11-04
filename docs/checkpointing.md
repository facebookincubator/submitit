# Checkpointing

When a job is preempted, or `signal_delay_s` before its time limit (see [`SlurmExecutor`](#submitit.SlurmExecutor)), it receives a signal and submitit requeues it:

- a callable without `__submitit_checkpoint__` method restarts from scratch, after preemption only;
- a callable with it is requeued with the submission this method returns, after preemption or timeout (at most `slurm_max_num_timeout` timeouts, default 3).

A signal received after min(80%, time limit - 10 minutes) of the time limit counts as a timeout.
Only the first task of a [multi-task job](examples.md#multi-task-jobs) calls `__submitit_checkpoint__`.
Preemptions are signaled only if the cluster sets `SlurmctldParameters=preempt_send_user_signal` (check with `scontrol show config | grep SlurmctldParameters`).

## Resume from the current state

`submitit.helpers.Checkpointable` implements `__submitit_checkpoint__` by requeuing the callable in its current state, with the same arguments:

```python
import time

import submitit


class Counter(submitit.helpers.Checkpointable):
    def __init__(self) -> None:
        self.count = 0  # saved when the job is requeued

    def __call__(self, until: int) -> int:
        while self.count < until:  # resumes from the saved count
            time.sleep(1)
            self.count += 1
        return self.count


executor = submitit.AutoExecutor(folder="logs/%j", slurm_max_num_timeout=5)
executor.update_parameters(timeout_min=60, slurm_partition="<your_partition>")
job = executor.submit(Counter(), 7200)
```

See also `submitit.helpers.FunctionSequence`, which runs several calls in one job and does not rerun finished calls after requeuing.

## Customize the resubmission

`__submitit_checkpoint__` receives the arguments of `__call__` and returns a `submitit.helpers.DelayedSubmission`, built like `executor.submit` arguments, or `None` to end the job (`job.result()` then raises an `UncompletedJobError`).
For instance, to save a model to a file rather than pickling it with the callable:

```python
from pathlib import Path

import submitit


class Training:
    def __init__(self) -> None:
        self.model = None

    def __call__(self, checkpoint_path: str) -> None:
        if Path(checkpoint_path).exists():
            self.model = ...  # load the model from checkpoint_path
        else:
            self.model = ...  # initialize the model
        ...  # train the model

    def __submitit_checkpoint__(self, checkpoint_path: str) -> submitit.helpers.DelayedSubmission:
        ...  # save self.model to checkpoint_path
        return submitit.helpers.DelayedSubmission(Training(), checkpoint_path)
```

## Test checkpointing

- On Slurm, `job._interrupt()` simulates a preemption, and `job._interrupt(timeout=True)` a timeout.
- Locally, run the job with `cluster="local"` and `timeout_min=1`: the local executor requeues timed-out jobs the same way.
