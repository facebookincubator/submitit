# PBS Examples

These examples show how to use `submitthem` with PBS (Portable Batch System), another common scheduler on HPC clusters.

## Files

### simple_pbs.py

Basic PBS example showing:
- AutoExecutor configuration for PBS
- Node allocation
- CPU and memory configuration
- Wall time management
- Simple job submission

**Run it:**
```bash
python simple_pbs.py
```

**Key PBS parameters used:**
- `time` - Wall time limit in minutes
- `nodes` - Number of nodes
- `cpus_per_task` - CPUs per task
- `mem_gb` - Memory per node in GB

### array_jobs_pbs.py

Advanced example showing:
- Submitting multiple similar jobs
- PBS-specific array job handling
- Efficient result collection
- Job monitoring and status tracking
- Performance summary

**Run it:**
```bash
python array_jobs_pbs.py
```

**Key features:**
- Demonstrates submitting 50 jobs
- Shows batch result collection
- Includes timeout management
- Provides execution summary

## PBS Configuration

### Common Parameters

```python
executor = submitthem.AutoExecutor(folder="./jobs")
executor.update_parameters(
    time=60,              # Wall time in minutes
    cpus_per_task=4,      # CPUs per task
    mem_gb=8,             # Memory per node in GB
    nodes=1,              # Number of nodes
    job_name="my_job",    # Job name in queue
)
```

### PBS-Specific Options

```python
# Use update_parameters() with PBS keywords:
executor.update_parameters(
    nodes=2,                    # Number of nodes
    cpus_per_task=8,           # CPUs per node
    mem_gb=32,                 # Memory per node
    select="1:ncpus=8:mem=32gb",  # Raw select clause
)
```

### Resource Selection Syntax

PBS uses a `select` clause for complex resource requests:

```python
# Simple node request
executor.update_parameters(nodes=2)  # 2 nodes

# With specific CPU/memory on each node
executor.update_parameters(
    select="2:ncpus=8:mem=32gb"  # 2 nodes, 8 CPUs, 32GB each
)
```

## Checking PBS Status

Before running examples, verify PBS is available:

```bash
# Check PBS availability
qstat

# List all jobs
qstat -a

# Check specific job
qstat -f <job_id>

# Submit a test job
qsub -l select=1:ncpus=1 -N test_job -- /bin/echo "Hello PBS"

# Delete a job
qdel <job_id>

# Check queue info
qstat -B  # Batch servers
qstat -q  # Queues
```

## Important PBS Concepts

### Job IDs
- PBS assigns job IDs automatically (usually numeric)
- Access via `job.job_id`
- Use `qstat -f <job_id>` to check detailed status

### Job States
- `Q` - Queued
- `R` - Running
- `H` - Held
- `S` - Suspended
- `C` - Completed
- `X` - Exiting
- `U` - Unknown

### Resource Selection
PBS uses `-l select=` clause:
```bash
# 1 node with 4 CPUs, 16GB memory
#PBS -l select=1:ncpus=4:mem=16gb

# 2 nodes with 8 CPUs each
#PBS -l select=2:ncpus=8

# Mixed resources
#PBS -l select=1:ncpus=4:mem=16gb:gpu=1
```

### Wall Time
Specify as `HH:MM:SS`:
```python
executor.update_parameters(time=60)  # 60 minutes = 01:00:00
```

### Wall Time Limits

Check your queue's limits:

```bash
qstat -q              # Show queues
qstat -q <queue_name> # Show specific queue details
```

## Tips and Tricks

1. **Debugging PBS issues**:
   ```bash
   # Check job script that was generated
   cat jobs/job_<id>/pbs_script.sh
   
   # Check job output
   cat jobs/job_<id>/output.log
   
   # Check detailed job info
   qstat -f <job_id>
   ```

2. **Testing before full submission**:
   - Start with `time=10` (10 minute timeout)
   - Use `nodes=1` to start
   - Scale up once it works

3. **Efficient job submission**:
   - Use array jobs for similar tasks
   - Monitor queue load
   - Batch similar resource requests

4. **Resource estimation**:
   - Monitor first run job details
   - Adjust parameters for next runs
   - Account for I/O and startup overhead

## Example Workflow

```python
import submitthem

# Function to submit
def my_computation(task_id):
    # Your code here
    return results

# Create executor
executor = submitthem.AutoExecutor(folder="./compute_jobs")

# Configure for PBS
executor.update_parameters(
    time=30,
    nodes=1,
    cpus_per_task=4,
    mem_gb=16,
)

# Submit jobs
jobs = [executor.submit(my_computation, i) for i in range(20)]

# Wait and collect results
results = [job.result(timeout=1800) for job in jobs]
```

## PBS vs SLURM

| Feature | PBS | SLURM |
|---------|-----|-------|
| Time limit | Minutes | Minutes |
| Memory | Per node | Per node or per CPU |
| Job arrays | `-J` | `--array=` |
| Resource select | `-l select=` | `-c`, `--mem=` |
| GPU request | In select clause | `--gres=gpu:` |
| Job query | `qstat` | `squeue` |

## Troubleshooting

### "qstat: command not found"
- PBS is not installed or not in PATH
- Try: `module load pbs` (on some clusters)
- Contact your cluster administrator

### Jobs stuck in queue
- Check queue limits: `qstat -q`
- Check node availability: `pbsnodes -a`
- Try smaller resource request

### Memory or resource error
- Increase `mem_gb` or adjust node count
- Check available resources: `pbsnodes -a`
- Review select clause syntax

### Job timeout
- Increase `time` parameter (in minutes)
- Profile code to find bottleneck
- Check for I/O bottlenecks

## Performance Optimization

1. **Resource efficiency**: 
   - Request only what you need
   - Monitor actual usage

2. **Queue strategy**:
   - Understand queue priorities
   - Use shortest queue for small jobs
   - Batch related jobs together

3. **I/O optimization**:
   - Use local scratch space when available
   - Minimize network I/O
   - Pre-stage data if possible

## Next Steps

- See [SLURM examples](../slurm/) for SLURM clusters
- See [local examples](../local/) for testing without PBS
- Read [PBS documentation](https://www.pbsworks.com//)
