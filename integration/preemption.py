# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""Preemption tests, need to be run on a an actual cluster"""
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import submitit
from submitit import AutoExecutor, Job
from submitit.core import test_core

FILE = Path(__file__)
LOGS = FILE.parent / "logs" / f"{FILE.stem}_log"

log = logging.getLogger("preemption_main")
formatter = logging.Formatter("%(name)s %(levelname)s (%(asctime)s) - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
log.setLevel(logging.INFO)
log.addHandler(handler)


def clock(partition: str, duration: int):
    log = logging.getLogger(f"preemption_{partition}")
    tick_tack = ["tick", "tack"]
    try:
        for minute in range(duration - 5):
            log.info(tick_tack[minute % 2])
            time.sleep(60)
        logging.warning("*** Exited peacefully ***")
        return duration
    except:
        logging.warning(f"!!! Interrupted on: {datetime.now().isoformat()}")
        raise


def pascal_job(partition: str, timeout_min: int, node: str = "") -> Job:
    """Submit a job with specific constraint that we can preempt deterministically."""
    ex = submitit.AutoExecutor(folder=LOGS, slurm_max_num_timeout=1)
    ex.update_parameters(
        name=f"submitit_preemption_{partition}",
        timeout_min=timeout_min,
        mem_gb=7,
        slurm_constraint="pascal",
        slurm_comment="submitit integration test",
        slurm_partition=partition,
        # pascal nodes have 80 cpus.
        # By requesting 50 we now that their can be only one such job with this property.
        cpus_per_task=50,
        slurm_additional_parameters={},
    )
    if node:
        ex.update_parameters(slurm_additional_parameters={"nodelist": node})

    return ex.submit(clock, partition, timeout_min)


def wait_job_is_running(job: Job) -> None:
    while job.state in ("UNKNOWN", "PENDING"):
        log.info(f"{job} is not RUNNING")
        time.sleep(60)


def preemption():
    job = pascal_job("learnfair", timeout_min=2 * 60)
    log.info(f"Scheduled {job}, {job.paths.stdout}")
    # log.info(job.paths.submission_file.read_text())

    wait_job_is_running(job)
    node = job.get_info()["NodeList"]
    log.info(f"{job} ({job.state}) is runnning on {node} !")
    # Schedule another pascal job on the same node, whith high priority
    priority_job = pascal_job("dev", timeout_min=15, node=node)
    log.info(f"Schedule {priority_job} ({job.state}) on {node} with high priority.")
    wait_job_is_running(priority_job)

    # if priority_job is running, then job should have been preempted
    learfair_stderr = job.stderr()
    assert learfair_stderr is not None, job.paths.stderr

    log.info(
        f"Job {priority_job} ({priority_job.state}) started, "
        f"job {job} ({job.state}) should have been preempted: {learfair_stderr}"
    )
    interruptions = [l for l in learfair_stderr.splitlines() if "Interrupted" in l]
    assert len(interruptions) == 1, interruptions
    assert job.state in ("PENDING"), job.state

    interrupted_ts = interruptions[0].split("!!! Interrupted on: ")[-1]
    interrupted = datetime.fromisoformat(interrupted_ts)

    priority_job.result()
    print("Preemption test succeeded ✅")


def main():
    log.info("Hello !")
    if LOGS.exists():
        log.info(f"Cleaning up log folder: {LOGS}")
        shutil.rmtree(str(LOGS))

    preemption()


if __name__ == "__main__":
    main()
