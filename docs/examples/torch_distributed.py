#!/usr/bin/env python

import os
import sys
import time

import submitit
import torch


NUM_NODES = 2
NUM_TASKS_PER_NODE = 8


NUM_CPUS_PER_TASK = 1
PARTITION = "devlab"
LOGS_DIR = "logs"


def print_env():
    for key in sorted(os.environ.keys()):
        if not (key.startswith("SLURM_") or
                key.startswith("SUBMITIT_") or
                key in ("MASTER_ADDR", "MASTER_PORT",
                        "RANK", "WORLD_SIZE",
                        "LOCAL_RANK", "LOCAL_WORLD_SIZE")):
            continue
        value = os.environ[key]
        print(f"{key}={value}")


class Task:
    def __call__(self):
        #print_env()
        print("exporting PyTorch distributed environment variables")
        job_env = submitit.JobEnvironment()
        params = submitit.helpers.export_torch_distributed_env(job_env)
        print(f"master: {params.master_addr}:{params.master_port}")
        print(f"rank: {params.rank}")
        print(f"world size: {params.world_size}")
        print(f"local rank: {params.local_rank}")
        print(f"local world size: {params.local_world_size}")
        #print_env()

        # Using env:// initialization method
        backend = torch.distributed.Backend.NCCL
        torch.distributed.init_process_group(backend=backend)
        assert params.rank == torch.distributed.get_rank()
        assert params.world_size == torch.distributed.get_world_size()

        # Actual task / computation
        device = torch.device("cuda", params.local_rank)
        result = params.rank * torch.ones(1).cuda(device=device)

        time.sleep(120)

        torch.distributed.all_reduce(result)
        if params.rank == 0:
            print(result)

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


def main():
    executor = submitit.AutoExecutor(folder=LOGS_DIR)
    executor.update_parameters(
        nodes=NUM_NODES,
        gpus_per_node=NUM_TASKS_PER_NODE,
        tasks_per_node=NUM_TASKS_PER_NODE,
        cpus_per_task=NUM_CPUS_PER_TASK,
        slurm_partition=PARTITION,
    )
    task = Task()
    job = executor.submit(task)
    submitit.helpers.monitor_jobs([job])
    return 0


if __name__ == "__main__":
    sys.exit(main())
