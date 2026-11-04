# Parameters

Resources are set with `executor.update_parameters(...)`, which applies to all later submissions of the executor and can be called several times.
A misspelled parameter raises a `NameError` listing the valid ones.

## Generic parameters

These parameters work on every executor and are converted to the cluster's own options:

| Parameter | Slurm option | Notes |
|---|---|---|
| `name` | `--job-name` | default: `submitit` |
| `timeout_min` | `--time` | in minutes, **default: 5 on Slurm, 2 locally** |
| `mem_gb` | `--mem` | non-integer values are converted to MB |
| `nodes` | `--nodes` | default: 1 |
| `tasks_per_node` | `--ntasks-per-node` | each task runs the function, see [Multi-tasks jobs](examples.md#multi-tasks-jobs) |
| `cpus_per_task` | `--cpus-per-task` | |
| `gpus_per_node` | `--gpus-per-node` | |
| `stderr_to_stdout` | no `--error` | writes stderr to the stdout log |

## Slurm parameters

Slurm-specific parameters are prefixed by `slurm_`, and ignored by other executors so that the same code also runs locally:

```python
import submitit

executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(
    name="train",
    timeout_min=24 * 60,
    gpus_per_node=8,
    tasks_per_node=8,  # one task per GPU
    cpus_per_task=10,
    mem_gb=400,
    slurm_partition="learn",
    slurm_account="my_account",
    slurm_qos="high",
    slurm_constraint="volta32gb",
)
```

A prefixed generic parameter (e.g. `slurm_timeout_min=60`) overrides the generic one on that cluster only.

Parameters written as the sbatch option of the same name, with `_` replaced by `-`:
`partition`, `account`, `qos`, `constraint`, `exclude`, `nodelist`, `gres`, `gpus_per_task`, `cpus_per_gpu`, `mem_per_gpu`, `mem_per_cpu`, `exclusive`, `dependency`, `comment`, `mail_type`, `mail_user`, and `wckey` (default: `submitit`, set `slurm_wckey=""` if your cluster rejects it).
See the [sbatch documentation](https://slurm.schedmd.com/sbatch.html) for their meaning.

Parameters specific to submitit:

| Parameter | Default | Effect |
|---|---|---|
| `array_parallelism` | 256 | maximum number of tasks of a job array running at the same time |
| `signal_delay_s` | 90 | seconds before the time limit at which the job is signaled, to [checkpoint](checkpointing.md) |
| `setup` | | shell commands run in the sbatch script before the job |
| `teardown` | | shell commands run in the sbatch script after the job |
| `srun_args` | | extra arguments of the `srun` call running the job |
| `use_srun` | `True` | run the job through `srun` in the sbatch script, or directly if `False` |
| `additional_parameters` | | sbatch options not supported above, overriding any other value |

```python
executor.update_parameters(
    slurm_setup=["module load cuda/12.1", "export NCCL_DEBUG=INFO"],
    slurm_additional_parameters={"reservation": "my_reservation"},
)
```

## Executor arguments

Some options are set when creating the executor, with the same prefix:

```python
executor = submitit.AutoExecutor(
    folder="logs/%j",
    slurm_python="/path/to/env/bin/python",
    slurm_max_num_timeout=5,
)
```

- `slurm_python`: command launching Python in the job, e.g. a container (`"singularity exec image.sif python"`); default: the current interpreter.
- `slurm_max_num_timeout` (default: 3): how many times a [checkpointable](checkpointing.md) job is requeued after a timeout.
- `slurm_max_pickle_size_gb` (default: 1.0): maximum size of the pickled function and arguments, as large pickles overload the shared filesystem.

## Local executor

The local executor uses `timeout_min`, `tasks_per_node`, `stderr_to_stdout` and `gpus_per_node`, requires `nodes=1` and ignores the other generic parameters.
`gpus_per_node=n` sets `CUDA_VISIBLE_DEVICES` to the first `n` GPUs: GPUs are not shared out between concurrent jobs.
`local_setup` runs shell commands before the job, like `slurm_setup`.
