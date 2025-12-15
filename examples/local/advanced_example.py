#!/usr/bin/env python
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
    )

    print("Local executor configuration:")
    print(f"  Executor type: LocalExecutor")
    print(f"  Folder: ./local_jobs")
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
    import time
    start_time = time.time()
    results = []
    for job in jobs:
        job_start = time.time()
        result = job.result()
        job_elapsed = time.time() - job_start
        total_elapsed = time.time() - start_time
        results.append(result)
        print(f"[{total_elapsed:6.1f}s] Job {job.job_id}: {result} (waited {job_elapsed:5.1f}s)")

    print(f"\nProcessed {len(results)} datasets")


if __name__ == "__main__":
    main()
