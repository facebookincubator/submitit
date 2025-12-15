#!/usr/bin/env python3
"""
Advanced example using SLURM with job arrays and batch processing.

This demonstrates how to submit job arrays and use batch mode
for efficient cluster utilization.
"""

import submitthem


def process_dataset(dataset_id: int, batch_size: int = 1000) -> dict:
    """
    Process a dataset with SLURM.

    Args:
        dataset_id: ID of dataset to process
        batch_size: Size of batches to process

    Returns:
        Processing results
    """
    total = sum(i for i in range(batch_size))
    return {
        "dataset_id": dataset_id,
        "processed_items": batch_size,
        "checksum": total,
    }


def main():
    """Submit a job array to SLURM."""
    executor = submitthem.AutoExecutor(
        folder="./slurm_jobs_advanced",
        cluster="slurm",
    )

    # Update SLURM-specific parameters
    executor.update_parameters(
        time=30,  # 30 minutes
        ntasks=8,  # 8 parallel tasks
        mem_gb=16,  # 16 GB memory
        gpus_per_node=2,  # 2 GPUs per node
    )

    # Submit multiple jobs in batch mode for efficiency
    print("Submitting jobs in batch mode...")
    with executor.batch():
        jobs = []
        for dataset_id in range(20):
            job = executor.submit(process_dataset, dataset_id, batch_size=5000)
            jobs.append(job)
        print(f"Submitted {len(jobs)} jobs in batch")

    # Collect results
    print("\nCollecting results...")
    import time
    start_time = time.time()
    results = []
    for i, job in enumerate(jobs):
        job_start = time.time()
        try:
            result = job.result()
            job_elapsed = time.time() - job_start
            total_elapsed = time.time() - start_time
            results.append(result)
            print(f"[{total_elapsed:6.1f}s] Dataset {result['dataset_id']}: processed {result['processed_items']} items (waited {job_elapsed:5.1f}s)")
        except Exception as e:
            job_elapsed = time.time() - job_start
            total_elapsed = time.time() - start_time
            print(f"[{total_elapsed:6.1f}s] Job {i} failed after {job_elapsed:5.1f}s: {e}")

    print(f"\nSuccessfully processed {len(results)} datasets")


if __name__ == "__main__":
    main()
