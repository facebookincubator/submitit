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
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        dist_env.export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        #print_env()

        # Using env:// initialization method
        backend = torch.distributed.Backend.NCCL
        torch.distributed.init_process_group(backend=backend)
        assert dist_env.rank == torch.distributed.get_rank()
        assert dist_env.world_size == torch.distributed.get_world_size()

        # Actual task / computation
        device = torch.device("cuda", dist_env.local_rank)
        result = dist_env.rank * torch.ones(1).cuda(device=device)

        time.sleep(120)

        torch.distributed.all_reduce(result)
        if dist_env.rank == 0:
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
