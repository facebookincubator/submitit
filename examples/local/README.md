# Local Execution Examples

These examples demonstrate how to use `submitthem` with local job execution. This is ideal for:

- **Testing and debugging** your job submission code
- **Development** before deploying to a cluster
- **Small-scale parallel processing** on your local machine
- **Understanding** how submitthem works

## Files

### simple_job.py

The simplest example showing:
- Basic job submission to local executor
- Simple function execution
- Result retrieval

**Run it:**
```bash
python simple_job.py
```

**Key features:**
- Uses `LocalExecutor` via `AutoExecutor`
- Simple data processing function
- Demonstrates basic workflow

### advanced_example.py

A more complete example showing:
- Advanced parameter configuration
- Data processing pipeline
- Job monitoring
- Result aggregation
- Error handling

**Run it:**
```bash
python advanced_example.py
```

**Key features:**
- Multiple job submissions
- Batch result collection
- Configuration management
- Status tracking

## Configuration

### Available Parameters for Local Execution

```python
executor = submitthem.AutoExecutor(folder="./jobs")
executor.update_parameters(
    # Local executor typically ignores cluster-specific parameters
    # but you can still set a job name
    job_name="my_job"
)
```

Most cluster-specific parameters (time, cpus, memory, etc.) are ignored by the local executor, but they won't cause errors if you set them.

## Tips

1. **Job Directory**: Jobs are saved in the folder you specify. This includes:
   - Job metadata
   - stdout/stderr
   - Pickled function and arguments
   - Results

2. **Debugging**: Check the job directory structure to understand:
   - How functions are serialized
   - Where logs are stored
   - How results are saved

3. **Parallel Execution**: The local executor can run jobs in parallel using multiprocessing. Check the implementation for `n_workers` parameter.

4. **Testing**: Use local execution to test your code before submitting to a real cluster:
   ```python
   # Start with local
   executor = submitthem.LocalExecutor()
   
   # Then switch to cluster
   executor = submitthem.SlurmExecutor()
   # Same code works!
   ```

## Performance Considerations

- Local execution is single-machine only
- Useful for small datasets and testing
- Not suitable for large-scale parallel workloads
- For real parallel processing, use SLURM or PBS examples

## Next Steps

Once you understand how these examples work, move on to:
- [SLURM examples](../slurm/) for cluster computing
- [PBS examples](../pbs/) for PBS clusters
