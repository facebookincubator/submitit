#!/usr/bin/env python3
"""
Example of submitting jobs to a PBS cluster using submitthem.

This script demonstrates how to use the AutoExecutor to automatically
detect and configure PBS, or explicitly use the PBSExecutor.
"""

import submitthem


def scientific_simulation(simulation_id: int, timesteps: int = 1000) -> dict:
    """
    A scientific simulation task.
    
    Args:
        simulation_id: ID of the simulation
        timesteps: Number of simulation timesteps
    
    Returns:
        Simulation results
    """
    # Simulate computation
    energy = 0.0
    for t in range(timesteps):
        energy += (t * 0.1) ** 2
    
    return {
        "simulation_id": simulation_id,
        "timesteps": timesteps,
        "total_energy": energy,
    }


def main():
    """Submit jobs to PBS cluster."""
    # Method 1: Use AutoExecutor (auto-detects the cluster)
    executor = submitthem.AutoExecutor(folder="./pbs_jobs")
    
    print(f"Executor type: {executor.cluster}")
    print()
    
    # Configure executor for PBS (if auto-detected)
    if executor.cluster == "pbs":
        executor.update_parameters(
            time=20,  # 20 minutes walltime
            nodes=2,  # Request 2 nodes
            cpus_per_task=4,  # 4 CPUs per task
            mem_gb=32,  # 32 GB memory per node
        )
        print("Configured PBS parameters:")
        print("  Walltime: 20 minutes")
        print("  Nodes: 2")
        print("  CPUs per task: 4")
        print("  Memory: 32 GB per node")
        print()
    
    # Submit jobs
    jobs = []
    simulation_ids = range(10)
    
    for sim_id in simulation_ids:
        job = executor.submit(scientific_simulation, sim_id, timesteps=5000)
        jobs.append(job)
        print(f"Submitted job {job.job_id}: simulation {sim_id}")
    
    # Monitor and collect results
    print("\nWaiting for simulations to complete...")
    results = []
    for i, job in enumerate(jobs):
        try:
            result = job.result(timeout=600)  # 10 minute timeout
            results.append(result)
            print(f"Job {job.job_id} (sim {result['simulation_id']}): "
                  f"energy={result['total_energy']:.2f}")
        except Exception as e:
            print(f"Job {job.job_id} failed: {e}")
    
    print(f"\nCompleted {len(results)}/{len(jobs)} simulations")


if __name__ == "__main__":
    main()
