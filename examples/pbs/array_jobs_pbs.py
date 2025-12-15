#!/usr/bin/env python3
"""
Example of submitting PBS array jobs with submitthem.

This demonstrates how to submit multiple jobs efficiently using array jobs,
which is important for submitting many similar jobs to a cluster.
"""

import submitthem


def process_dataset(dataset_id: int, num_chunks: int = 100) -> dict:
    """
    Process a dataset in chunks.

    Args:
        dataset_id: ID of the dataset to process
        num_chunks: Number of chunks to process

    Returns:
        Processing results
    """
    total_processed = 0
    errors = 0

    for chunk_idx in range(num_chunks):
        try:
            # Simulate processing
            chunk_size = 1000 + chunk_idx * 10
            total_processed += chunk_size
        except Exception:
            errors += 1

    return {
        "dataset_id": dataset_id,
        "total_processed": total_processed,
        "errors": errors,
        "success": errors == 0,
    }


def main():
    """Submit array jobs to PBS cluster for parallel processing."""
    # Use AutoExecutor
    executor = submitthem.AutoExecutor(
        folder="./pbs_array_jobs",
    )

    print(f"Executor: {executor.cluster}")
    print()

    # Configure for PBS
    if executor.cluster == "pbs":
        executor.update_parameters(
            pbs_time=60,  # 1 hour per job
            cpus_per_task=8,  # 8 CPUs per task
            mem_gb=16,  # 16 GB memory
        )
        print("PBS Configuration:")
        print("  Walltime: 60 minutes")
        print("  CPUs: 8")
        print("  Memory: 16 GB")
        print()

    # Method 1: Submit jobs individually (simpler, but slower)
    NB_JOBS = 2
    print(f"Submitting {NB_JOBS} dataset processing jobs...")
    jobs = []
    for dataset_id in range(NB_JOBS):
        job = executor.submit(process_dataset, dataset_id)
        jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")
    print()

    # Method 2: Batch mode (more efficient - submits multiple jobs in one go)
    # This can be more efficient if your scheduler supports batch operations
    print("Collecting results with batch monitoring...")
    import time
    start_time = time.time()

    results = []
    failed_jobs = []

    for job in jobs:
        job_start = time.time()
        try:
            result = job.result()
            job_elapsed = time.time() - job_start
            total_elapsed = time.time() - start_time
            results.append(result)
            if result["success"]:
                print(f"[{total_elapsed:6.1f}s] Job {job.job_id}: processed {result['total_processed']} items (waited {job_elapsed:5.1f}s)")
            else:
                print(f"[{total_elapsed:6.1f}s] Job {job.job_id}: completed with {result['errors']} errors (waited {job_elapsed:5.1f}s)")
        except Exception as e:
            job_elapsed = time.time() - job_start
            total_elapsed = time.time() - start_time
            print(f"[{total_elapsed:6.1f}s] Job {job.job_id}: failed after {job_elapsed:5.1f}s with error {e}")
            failed_jobs.append(job.job_id)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total jobs: {len(jobs)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_jobs)}")

    if results:
        total_items = sum(r["total_processed"] for r in results)
        print(f"Total items processed: {total_items}")

    if failed_jobs:
        print(f"Failed job IDs: {failed_jobs}")


if __name__ == "__main__":
    main()
