#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import sys
import time

import torch

import submitit

NUM_NODES = 2
NUM_TASKS_PER_NODE = 8


NUM_CPUS_PER_TASK = 1
PARTITION = "<your_partition>"
LOGS_DIR = "logs"


class Task:
    def __call__(self):
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")

        # Using the (default) env:// initialization method
        torch.distributed.init_process_group(backend="nccl")
        assert dist_env.rank == torch.distributed.get_rank()
        assert dist_env.world_size == torch.distributed.get_world_size()

        # Actual task / computation
        tensor = dist_env.rank * torch.ones(1).cuda()

        time.sleep(120)

        torch.distributed.all_reduce(tensor)
        if dist_env.rank == 0:
            result = list(tensor)
            print(result)
            return result

    def __submitit_checkpoint__(self):
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
    print(job.results()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
