#!/usr/bin/env python3
"""
Example of submitting jobs to a SLURM cluster using submitthem.

This script demonstrates how to use the AutoExecutor to automatically
detect and configure SLURM, or explicitly use the SlurmExecutor.
"""

import submitthem


def compute_intensive_task(n: int) -> int:
    """
    A compute-intensive task that benefits from cluster execution.
    
    Args:
        n: Problem size
    
    Returns:
        Result of computation
    """
    # Simulate computation
    result = 0
    for i in range(n):
        result += i**2
    return result


def main():
    """Submit jobs to SLURM cluster."""
    # Method 1: Use AutoExecutor (auto-detects the cluster)
    executor = submitthem.AutoExecutor(folder="./slurm_jobs")
    
    print(f"Executor type: {executor.cluster}")
    print()
    
    # Configure executor for SLURM (if auto-detected)
    if executor.cluster == "slurm":
        executor.update_parameters(
            time=10,  # 10 minutes
            gpus_per_node=1,  # Request 1 GPU per node
            ntasks=4,  # Request 4 tasks
            mem_gb=8,  # Request 8 GB memory
        )
        print("Configured SLURM parameters:")
        print(f"  Time: 10 minutes")
        print(f"  GPUs per node: 1")
        print(f"  Tasks: 4")
        print(f"  Memory: 8 GB")
        print()
    
    # Submit jobs
    jobs = []
    problem_sizes = [1000, 5000, 10000, 50000]
    
    for size in problem_sizes:
        job = executor.submit(compute_intensive_task, size)
        jobs.append(job)
        print(f"Submitted job {job.job_id}: computing with n={size}")
    
    # Monitor jobs
    print("\nWaiting for jobs to complete...")
    results = []
    for i, job in enumerate(jobs):
        try:
            result = job.result(timeout=300)  # 5 minute timeout
            results.append(result)
            print(f"Job {job.job_id} (n={problem_sizes[i]}): {result}")
        except submitthem.UncompletedJobError:
            print(f"Job {job.job_id} did not complete in time")
    
    print(f"\nCompleted {len(results)}/{len(jobs)} jobs")


if __name__ == "__main__":
    main()
