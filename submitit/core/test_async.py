# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import asyncio
from pathlib import Path

import pytest

from . import submission, utils
from .test_core import FakeExecutor, _three_time


@pytest.mark.asyncio
async def test_result(tmp_path: Path) -> None:
    event_loop = asyncio.get_running_loop()
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    result_task = event_loop.create_task(job.awaitable().result())
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        submission.process_job(folder=job.paths.folder)
    result = await result_task
    assert result == 24


@pytest.mark.asyncio
async def test_results_single(tmp_path: Path) -> None:
    event_loop = asyncio.get_running_loop()
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    result_task = event_loop.create_task(job.awaitable().results())
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        submission.process_job(folder=job.paths.folder)
    result = await result_task
    assert result == [24]


@pytest.mark.asyncio
@pytest.mark.parametrize("num_tasks", [1, 2])
async def test_results_ascompleted(tmp_path: Path, num_tasks: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeExecutor, "_num_tasks", lambda self: num_tasks)
    executor = FakeExecutor(folder=tmp_path)
    job = executor.submit(_three_time, 8)
    for task in range(num_tasks):
        env = {"SLURM_JOB_ID": str(job.job_id), "SLURM_PROCID": str(task), "SLURM_NTASKS": str(num_tasks)}
        with utils.environment_variables(_TEST_CLUSTER_="slurm", **env):
            submission.process_job(folder=job.paths.folder)
    assert [await aws for aws in job.awaitable().results_as_completed()] == [24] * num_tasks
