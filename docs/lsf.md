# LSF Support

Submitit supports IBM Spectrum LSF (Load Sharing Facility) as a backend for job submission. This document describes LSF-specific configuration and usage.

## Basic Usage

To use LSF with submitit, simply use `AutoExecutor` which will automatically detect if you're on an LSF cluster:

```python
import submitit

def my_function(x, y):
    return x + y

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=60, lsf_queue="normal")
job = executor.submit(my_function, 5, 7)
print(job.result())  # 12
```

You can also explicitly request LSF:

```python
executor = submitit.AutoExecutor(folder="logs", cluster="lsf")
```

Or use `LsfExecutor` directly:

```python
executor = submitit.LsfExecutor(folder="logs")
```

## Parameters

LSF-specific parameters should be prefixed with `lsf_` when using `AutoExecutor`. When using `LsfExecutor` directly, no prefix is needed.

### Common Parameters

| AutoExecutor Parameter | LsfExecutor Parameter | Description |
|------------------------|----------------------|-------------|
| `timeout_min` | `time` | Job time limit in minutes |
| `mem_gb` | `mem` | Memory limit in GB |
| `gpus_per_node` | `gpus_per_node` | Number of GPUs per node |
| `cpus_per_task` | `cpus_per_task` | Number of CPUs per task |
| `name` | `job_name` | Job name |
| `lsf_queue` | `queue` | LSF queue name |
| `lsf_setup` | `setup` | List of shell commands to run before the job |
| `lsf_teardown` | `teardown` | List of shell commands to run after the job |
| `lsf_array_parallelism` | `array_parallelism` | Max concurrent array jobs |
| `lsf_additional_parameters` | `additional_parameters` | Dict of additional `#BSUB` flags |

### Example with Multiple Parameters

```python
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(
    timeout_min=120,
    mem_gb=16,
    gpus_per_node=2,
    cpus_per_task=8,
    name="my_job",
    lsf_queue="gpu",
    lsf_setup=["module load cuda/11.0"],
    lsf_additional_parameters={"P": "my_project"},
)
```

## Job Arrays

Submitit supports LSF job arrays through `map_array()` or the batch API:

```python
executor = submitit.AutoExecutor(folder="logs", cluster="lsf")
executor.update_parameters(lsf_array_parallelism=10)  # max 10 concurrent jobs

# Using map_array
jobs = executor.map_array(my_function, [1, 2, 3], [4, 5, 6])

# Using batch API
with executor.batch():
    jobs = [executor.submit(my_function, i, i+1) for i in range(100)]
```

LSF arrays use 1-based indexing. The job IDs follow submitit's convention: `{array_job_id}_{array_index}`, where `array_index` starts at 1.

## Checkpointing and Requeue

Submitit supports checkpointing on LSF, which allows jobs to save their state before being killed due to timeout or preemption.

### Requirements

For checkpointing to work, your LSF cluster must be configured to send a warning signal before killing jobs. Submitit uses `SIGUSR2` by default. This is configured in the submission script via:

```
#BSUB -wt '90s'    # Warning time: 90 seconds before kill
#BSUB -wa 'USR2'   # Warning action: send SIGUSR2
```

The `signal_delay_s` parameter controls this warning time (default: 90 seconds).

### Using Checkpointable Classes

To enable checkpointing, your callable must implement a `checkpoint` method:

```python
import submitit

class MyTrainer(submitit.helpers.Checkpointable):
    def __init__(self):
        self.epoch = 0
        self.model = None
    
    def __call__(self, num_epochs):
        if self.model is None:
            self.model = create_model()
        
        for epoch in range(self.epoch, num_epochs):
            train_one_epoch(self.model)
            self.epoch = epoch + 1
        
        return self.model
    
    def checkpoint(self, num_epochs):
        # Save model to disk if needed
        return submitit.helpers.DelayedSubmission(self, num_epochs)

executor = submitit.AutoExecutor(folder="logs", cluster="lsf", lsf_max_num_timeout=3)
executor.update_parameters(timeout_min=60)
trainer = MyTrainer()
job = executor.submit(trainer, 100)
```

### Requeue Behavior

When a job receives a warning signal:
1. If the callable has a `checkpoint` method, it's called to save state
2. The job is requeued using `brequeue`
3. On restart, the callable resumes from the saved state

Jobs are requeued up to `max_num_timeout` times (default: 3) for timeouts. Preempted jobs are always requeued.

## Log Folder Requirements

**Important**: The log folder must be on a shared filesystem accessible from both the submission host and all compute nodes. Do not use `/tmp` or other local filesystems.

```python
# Good: shared filesystem
executor = submitit.AutoExecutor(folder="/shared/logs/my_experiment")

# Bad: local filesystem (will fail silently!)
executor = submitit.AutoExecutor(folder="/tmp/logs")
```

## Environment Variables

Inside an LSF job, the following environment variables are available:

| Variable | Description |
|----------|-------------|
| `LSB_JOBID` | LSF job ID |
| `LSB_HOSTS` | Space-separated list of allocated hosts |
| `LSB_JOBINDEX` | Array job index (if array job) |
| `SUBMITIT_EXECUTOR` | Set to `lsf` |
| `SUBMITIT_LSF_ARRAY_JOB_ID` | Parent array job ID (if array job) |
| `SUBMITIT_LSF_ARRAY_TASK_ID` | Array task index (if array job) |

Access these through `JobEnvironment`:

```python
import submitit

def my_job():
    env = submitit.JobEnvironment()
    print(f"Job ID: {env.job_id}")
    print(f"Hostname: {env.hostname}")
    if env.array_job_id:
        print(f"Array index: {env.array_task_id}")
```

## Differences from Slurm

| Feature | Slurm | LSF |
|---------|-------|-----|
| Submit command | `sbatch` | `bsub` (stdin) |
| Cancel command | `scancel` | `bkill` |
| Status command | `sacct` | `bjobs` |
| Requeue command | `scontrol requeue` | `brequeue` |
| Array syntax | `--array=0-N%P` | `-J "name[0-N]%P"` |
| Log placeholders | `%j`, `%a` | `%J`, `%I` |

## Troubleshooting

### Job fails without logs

- Ensure the log folder is on a shared filesystem
- Check that the folder has write permissions
- Verify LSF can reach the filesystem from compute nodes

### Checkpointing not working

- Verify your LSF cluster sends warning signals (`-wa` and `-wt` directives)
- Check that your callable properly implements `checkpoint()`
- Ensure `max_num_timeout > 0`

### Array jobs not running in parallel

- Check `array_parallelism` setting
- Verify your queue allows array jobs
- Check queue limits for concurrent jobs

### Cannot find `bsub`

- Ensure LSF is installed and in your `$PATH`
- Load the LSF module if needed: `module load lsf`

