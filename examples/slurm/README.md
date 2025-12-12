# SLURM Examples

These examples show how to use `submitthem` with SLURM (Slurm Workload Manager), the most common scheduler on HPC clusters.

## Files

### simple_slurm.py

Basic SLURM example showing:
- AutoExecutor configuration for SLURM
- GPU allocation
- Memory and CPU configuration
- Wall time management
- Simple job submission

**Run it:**
```bash
python simple_slurm.py
```

**Key SLURM parameters used:**
- `time` - Wall time limit in minutes
- `cpus_per_task` - CPUs per task
- `mem_gb` - Memory per node in GB
- `gpu` - Number of GPUs

### array_jobs.py

Advanced example showing:
- Submitting multiple similar jobs
- Batch job submission
- Array job handling
- Efficient result collection
- Job monitoring

**Run it:**
```bash
python array_jobs.py
```

**Key features:**
- Demonstrates submitting 100+ jobs efficiently
- Shows how to handle large job batches
- Includes timeout management
- Includes result aggregation

## SLURM Configuration

### Common Parameters

```python
executor = submitthem.AutoExecutor(folder="./jobs")
executor.update_parameters(
    time=60,              # Wall time in minutes
    cpus_per_task=4,      # CPUs per task
    mem_gb=8,             # Memory in GB
    gpu=1,                # Number of GPUs
    job_name="my_job",    # Job name in queue
)
```

### Additional SLURM Options

```python
# Use update_parameters() with SLURM-specific keywords:
executor.update_parameters(
    nodes=2,                    # Number of nodes
    cpus_per_task=8,           # CPUs per task
    mem_per_cpu_gb=4,          # Memory per CPU in GB
    exclusive=True,            # Exclusive node allocation
    constraint="gpu:v100",     # Node constraint (GPU type)
)
```

## Checking SLURM Status

Before running examples, verify SLURM is available:

```bash
# Check SLURM availability
sinfo

# Check cluster partitions
sinfo --partition=all

# Submit a test job
sbatch --wrap="echo 'Hello from SLURM'"

# Check running jobs
squeue

# Check job details
squeue -j <job_id> --long
```

## Important SLURM Concepts

### Job IDs
- SLURM assigns job IDs automatically
- Access via `job.job_id`
- Use `squeue -j <job_id>` to check status

### Array Jobs
- Efficiently submit many similar jobs
- Useful for parameter sweeps
- Example: `#SBATCH --array=0-99`

### GPU Selection
- Check available GPUs: `sinfo --gres=gpu`
- Specific GPU type: `--gres=gpu:v100:1`
- GPU memory: `--mem-per-gpu=16G`

### Memory Specification
- Per node: `--mem=32G`
- Per CPU: `--mem-per-cpu=8G`
- Not both together!

## Tips and Tricks

1. **Debugging SLURM issues**:
   ```bash
   # Check job script that was generated
   cat jobs/job_<id>/slurm_script.sh
   
   # Check job output
   cat jobs/job_<id>/output.log
   ```

2. **Testing before full submission**:
   - Start with `time=10` (10 minute timeout)
   - Use `cpus_per_task=1` to start
   - Scale up once it works

3. **Efficient job submission**:
   - Batch similar jobs together
   - Use array jobs for parameter sweeps
   - Monitor queue to avoid overloading

4. **Resource estimation**:
   - Monitor first run: `sstat -j <job_id>`
   - Adjust parameters for next runs
   - Account for startup overhead

## Example Workflow

```python
import submitthem

# Function to submit
def my_analysis(data_id):
    # Your code here
    return results

# Create executor
executor = submitthem.AutoExecutor(folder="./analysis_jobs")

# Configure for SLURM
executor.update_parameters(
    time=30,
    cpus_per_task=4,
    mem_gb=16,
)

# Submit jobs
jobs = [executor.submit(my_analysis, i) for i in range(10)]

# Wait and collect
results = [job.result(timeout=1800) for job in jobs]
```

## Troubleshooting

### "sbatch: not found"
- SLURM is not installed or not in PATH
- Try: `module load slurm` (on some clusters)

### Jobs stuck in queue
- Check: `squeue | grep <your_username>`
- May be waiting for resources or node availability
- Check partition limits: `sinfo`

### Memory error
- Increase `mem_gb` parameter
- Monitor actual usage: `sstat -j <job_id>`
- Consider memory-per-CPU: `--mem-per-cpu`

### Job timeout
- Increase `time` parameter (in minutes)
- Profile code to find bottleneck
- Consider parallelization

## Performance Optimization

1. **Minimize wall time**: Only request time you need
2. **Resource efficiency**: Don't over-allocate memory/CPUs
3. **I/O optimization**: Use local `/tmp` for I/O intensive jobs
4. **Batch operations**: Submit many jobs at once

## Next Steps

- See [PBS examples](../pbs/) for PBS clusters
- See [local examples](../local/) for testing without SLURM
- Read [SLURM documentation](https://slurm.schedmd.com/)
