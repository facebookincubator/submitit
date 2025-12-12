#!/usr/bin/env python3
"""
Advanced example of using the Local executor with parameter updates.

This example demonstrates how to configure the executor with custom parameters
and submit jobs with different computational requirements.
"""

import submitthem


def data_processing(data_id: int, num_iterations: int = 100) -> dict:
    """
    Simulate a data processing task.
    
    Args:
        data_id: Identifier for the data
        num_iterations: Number of iterations to perform
    
    Returns:
        Dictionary with processing results
    """
    result = sum(i**2 for i in range(num_iterations))
    return {
        "data_id": data_id,
        "iterations": num_iterations,
        "checksum": result,
    }


def main():
    """Submit data processing jobs with the local executor."""
    # Create executor with custom parameters
    executor = submitthem.LocalExecutor(
        folder="./local_jobs",
        local_num_threads=2,  # Use 2 parallel processes
    )
    
    print("Local executor configuration:")
    print(f"  Cluster: {executor.cluster}")
    print(f"  Parameters: {executor._executor.parameters}")
    print()
    
    # Submit jobs with different data IDs
    jobs = []
    data_ids = [100, 200, 300, 400]
    
    for data_id in data_ids:
        job = executor.submit(data_processing, data_id, num_iterations=1000)
        jobs.append(job)
        print(f"Submitted job {job.job_id}: processing data_id={data_id}")
    
    # Collect results as jobs complete
    print("\nCollecting results...")
    results = []
    for job in jobs:
        result = job.result()
        results.append(result)
        print(f"Job {job.job_id}: {result}")
    
    print(f"\nProcessed {len(results)} datasets")


if __name__ == "__main__":
    main()
