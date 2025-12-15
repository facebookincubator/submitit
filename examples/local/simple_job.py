#!/usr/bin/env python
"""
Simple example of submitting a job using the Local executor.

The local executor runs jobs in parallel processes on the local machine,
useful for testing and development without access to a cluster.
"""

import submitthem


def compute_task(x: int, y: int) -> int:
    """A simple computational task."""
    return x * y + x**2


def main():
    """Submit jobs using the local executor."""
    # Create a local executor
    executor = submitthem.LocalExecutor(folder="./local_jobs")

    # Submit multiple jobs
    jobs = []
    for i in range(5):
        job = executor.submit(compute_task, i, i + 1)
        jobs.append(job)
        print(f"Submitted job {job.job_id}: compute_task({i}, {i+1})")

    # Wait for all jobs to complete and collect results
    print("\nWaiting for jobs to complete...")
    import time
    start_time = time.time()
    results = []
    for i, job in enumerate(jobs):
        job_start = time.time()
        result = job.result()
        job_elapsed = time.time() - job_start
        total_elapsed = time.time() - start_time
        results.append(result)
        print(f"[{total_elapsed:6.1f}s] Job {job.job_id} completed with result: {result} (waited {job_elapsed:5.1f}s)")

    print(f"\nAll results: {results}")


if __name__ == "__main__":
    main()
