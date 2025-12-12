# submitthem Examples

This directory contains example scripts demonstrating how to use `submitthem` to submit and manage jobs on different cluster schedulers.

## Directory Structure

- **local/** - Examples for local job execution (useful for testing)
- **slurm/** - Examples for SLURM-based clusters
- **pbs/** - Examples for PBS-based clusters

## Quick Start

### 1. Local Execution (Testing)

For testing your job submission code locally without a real cluster:

```bash
cd local
python simple_job.py
```

This is useful for:
- Debugging job submission logic
- Testing your code before submitting to a real cluster
- Understanding how submitthem works

### 2. SLURM Cluster

For clusters using SLURM (Slurm Workload Manager):

```bash
cd slurm
python simple_slurm.py
```

### 3. PBS Cluster

For clusters using PBS (Portable Batch System):

```bash
cd pbs
python simple_pbs.py
```

## Common Patterns

### Basic Job Submission

All examples follow this pattern:

```python
import submitthem

def my_task(arg1, arg2):
    # Your computation here
    return result

# Create executor
executor = submitthem.AutoExecutor(folder="./jobs")

# Submit job
job = executor.submit(my_task, value1, value2)

# Wait for result
result = job.result(timeout=3600)  # 1 hour timeout
```

### Array/Batch Jobs

For submitting many similar jobs efficiently:

```python
jobs = []
for i in range(100):
    job = executor.submit(my_task, i)
    jobs.append(job)

# Collect results
for job in jobs:
    result = job.result()
```

### Configuring Scheduler Parameters

Use `update_parameters()` to configure cluster-specific settings:

```python
executor.update_parameters(
    time=60,           # Wall time in minutes
    cpus=4,            # Number of CPUs
    mem_gb=8,          # Memory in GB
    nodes=2,           # Number of nodes (SLURM)
)
```

## Example Details

### local/simple_job.py
Basic example showing:
- Simple function submission
- Job execution on local machine
- Result collection

### local/advanced_example.py
More complex example showing:
- Parameter configuration
- Data processing pipeline
- Error handling
- Result aggregation

### slurm/simple_slurm.py
SLURM-specific example showing:
- AutoExecutor with SLURM auto-detection
- GPU configuration
- Memory requirements
- Wall time management

### slurm/array_jobs.py
Advanced SLURM example showing:
- Submitting many jobs
- Array job parameters
- Batch result collection
- Job status monitoring

### pbs/simple_pbs.py
PBS-specific example showing:
- AutoExecutor for PBS
- Node configuration
- Resource allocation
- Result monitoring

### pbs/array_jobs_pbs.py
Advanced PBS example showing:
- Submitting job arrays
- PBS-specific resource requests
- Batch job monitoring
- Performance summary

## Choosing the Right Executor

Use `AutoExecutor` to automatically detect your cluster type:

```python
executor = submitthem.AutoExecutor(folder="./jobs")
```

If auto-detection fails or you want to be explicit:

```python
# For SLURM:
executor = submitthem.SlurmExecutor(folder="./jobs")

# For PBS:
executor = submitthem.PBSExecutor(folder="./jobs")

# For local:
executor = submitthem.LocalExecutor(folder="./jobs")
```

## Important Notes

1. **Job Function Requirements**:
   - Must be picklable (defined at module level)
   - Can accept any number of arguments
   - Should return serializable objects

2. **Resource Configuration**:
   - Check your cluster's resource limits
   - Use `executor.update_parameters()` to set proper values
   - Too many resources → longer queue wait time
   - Too few resources → job failure

3. **Job Directories**:
   - Jobs are saved in the folder you specify
   - Each job gets its own subdirectory
   - Clean up old jobs regularly

4. **Monitoring**:
   - Use `job.job_id` to get the cluster job ID
   - Use `job.result(timeout=...)` to wait for completion
   - Use `job.status()` to check job status

## Advanced Features

For more advanced features, see the main `docs/` directory:

- **checkpointing.md** - Save and resume long-running jobs
- **plugins.md** - Extend submitthem with custom schedulers
- **tips.md** - Performance optimization and best practices

## Troubleshooting

### Jobs not running
- Check cluster availability: `squeue` (SLURM) or `qstat` (PBS)
- Check job directory permissions
- Verify resource configuration

### Import errors
- Ensure `submitthem` is installed: `pip install submitthem`
- Check Python path and virtual environment

### Performance issues
- Don't submit too many jobs at once
- Use batch operations when available
- Check job logs in the job directory
