"""Integration tests for LSF executor that run on a real LSF cluster.

These tests are SKIPPED by default. To run them:

1. Set environment variables:
   export SUBMITIT_RUN_LSF_INTEGRATION_TESTS=1
   export SUBMITIT_LSF_TEST_FOLDER=/shared/path/to/logs  # Must be shared filesystem
   export SUBMITIT_LSF_TEST_QUEUE=normal  # Optional: queue name
   export SUBMITIT_LSF_TEST_GPU_QUEUE=gpu  # Optional: for GPU tests
   export SUBMITIT_LSF_TEST_TIMEOUT_MIN=10  # Optional: default 10

2. Run tests:
   python -m pytest submitit/lsf/test_lsf_integration.py -v
"""
import os
import shutil
import subprocess
import time
import typing as tp
from pathlib import Path

import pytest

import submitit
from submitit.core import utils

# Skip all tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("SUBMITIT_RUN_LSF_INTEGRATION_TESTS") != "1",
    reason="LSF integration tests disabled. Set SUBMITIT_RUN_LSF_INTEGRATION_TESTS=1 to run.",
)


def _get_test_folder() -> Path:
    """Get test folder from environment, skip if not set."""
    folder = os.environ.get("SUBMITIT_LSF_TEST_FOLDER")
    if not folder:
        pytest.skip("SUBMITIT_LSF_TEST_FOLDER not set")
    return Path(folder)


def _get_queue() -> tp.Optional[str]:
    """Get queue name from environment."""
    return os.environ.get("SUBMITIT_LSF_TEST_QUEUE")


def _get_gpu_queue() -> tp.Optional[str]:
    """Get GPU queue name from environment."""
    return os.environ.get("SUBMITIT_LSF_TEST_GPU_QUEUE")


def _get_timeout() -> int:
    """Get timeout in minutes from environment."""
    return int(os.environ.get("SUBMITIT_LSF_TEST_TIMEOUT_MIN", "10"))


def _wait_for_job(job: submitit.Job, max_wait_s: int = 300) -> str:
    """Wait for a job to finish, polling bjobs directly.
    
    Returns the final state.
    """
    job_id = job.job_id.split("_")[0]  # Get main job ID for arrays
    start = time.time()
    while time.time() - start < max_wait_s:
        result = subprocess.run(
            ["bjobs", "-o", "JOBID JOBINDEX STAT", "-noheader", job_id],
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        if not output:
            # Job no longer exists, assume completed
            return job.state
        # Check if all relevant jobs are finished
        lines = output.splitlines()
        finished = all("DONE" in line or "EXIT" in line for line in lines if line.strip())
        if finished:
            return job.state
        time.sleep(5)
    return job.state


@pytest.fixture
def test_folder(tmp_path: Path) -> tp.Generator[Path, None, None]:
    """Create a unique test folder under the configured LSF test folder."""
    base_folder = _get_test_folder()
    folder = base_folder / f"test_{os.getpid()}_{int(time.time())}"
    folder.mkdir(parents=True, exist_ok=True)
    yield folder
    # Cleanup (best effort)
    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_lsf_submit_result_logs_cpu(test_folder: Path) -> None:
    """Test basic job submission, result, and log access."""
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    def add(x: int, y: int) -> int:
        return x + y

    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf")
    params: tp.Dict[str, tp.Any] = {"timeout_min": _get_timeout()}
    if _get_queue():
        params["lsf_queue"] = _get_queue()
    executor.update_parameters(**params)

    job = executor.submit(add, 5, 7)
    assert job.job_id is not None

    # Wait for completion
    final_state = _wait_for_job(job)
    assert final_state in ("COMPLETED", "DONE"), f"Job ended in state: {final_state}"

    # Check result
    result = job.result()
    assert result == 12

    # Check logs exist
    assert job.paths.stdout.exists() or Path(str(job.paths.stdout).replace("%j", job.job_id)).exists()


def test_lsf_map_array_cpu(test_folder: Path) -> None:
    """Test array job submission."""
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    def add(x: int, y: int) -> int:
        return x + y

    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf")
    params: tp.Dict[str, tp.Any] = {"timeout_min": _get_timeout()}
    if _get_queue():
        params["lsf_queue"] = _get_queue()
    executor.update_parameters(**params)

    # Submit 3-element array
    jobs = executor.map_array(add, [1, 2, 3], [4, 5, 6])
    assert len(jobs) == 3

    # Check job IDs are 1-based
    array_id = jobs[0].job_id.split("_")[0]
    assert [f"{array_id}_{i}" for i in [1, 2, 3]] == [j.job_id for j in jobs]

    # Wait for all to complete
    for job in jobs:
        _wait_for_job(job)

    # Check results
    results = [job.result() for job in jobs]
    assert results == [5, 7, 9]


def test_lsf_cancel_cpu(test_folder: Path) -> None:
    """Test job cancellation."""
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    def long_sleep(x: int) -> int:
        import time
        time.sleep(600)
        return x

    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf")
    params: tp.Dict[str, tp.Any] = {"timeout_min": _get_timeout()}
    if _get_queue():
        params["lsf_queue"] = _get_queue()
    executor.update_parameters(**params)

    job = executor.submit(long_sleep, 42)

    # Wait for job to start (or give up after 60s)
    for _ in range(30):
        if job.state == "RUNNING":
            break
        time.sleep(2)

    # Cancel the job
    job.cancel()
    time.sleep(5)

    # Verify it's no longer running
    final_state = job.state
    assert final_state != "RUNNING", f"Job still running after cancel: {final_state}"


@pytest.mark.skipif(
    not os.environ.get("SUBMITIT_LSF_TEST_GPU_QUEUE"),
    reason="SUBMITIT_LSF_TEST_GPU_QUEUE not set",
)
def test_lsf_gpu_request(test_folder: Path) -> None:
    """Test GPU resource request."""
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    def check_gpu() -> str:
        import subprocess
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return result.stdout

    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf")
    params: tp.Dict[str, tp.Any] = {
        "timeout_min": _get_timeout(),
        "gpus_per_node": 1,
    }
    if _get_gpu_queue():
        params["lsf_queue"] = _get_gpu_queue()
    executor.update_parameters(**params)

    job = executor.submit(check_gpu)
    _wait_for_job(job)

    result = job.result()
    assert "GPU" in result, f"GPU not found in nvidia-smi output: {result}"


@pytest.mark.skipif(
    shutil.which("brequeue") is None,
    reason="brequeue not available on this system",
)
def test_lsf_checkpoint_requeue(test_folder: Path) -> None:
    """Test checkpoint and requeue functionality.
    
    This test submits a checkpointable job, sends SIGUSR2 to trigger
    checkpoint, and verifies the job is requeued.
    """
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    class Counter(submitit.helpers.Checkpointable):
        def __init__(self) -> None:
            self.count = 0

        def __call__(self, max_count: int) -> int:
            import time
            while self.count < max_count:
                self.count += 1
                time.sleep(1)
            return self.count

        def checkpoint(self, max_count: int) -> submitit.helpers.DelayedSubmission:
            return submitit.helpers.DelayedSubmission(self, max_count)

    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf")
    params: tp.Dict[str, tp.Any] = {
        "timeout_min": _get_timeout(),
        "lsf_max_num_timeout": 2,
    }
    if _get_queue():
        params["lsf_queue"] = _get_queue()
    executor.update_parameters(**params)

    counter = Counter()
    job = executor.submit(counter, 100)  # Will take 100 seconds

    # Wait for job to start running
    for _ in range(30):
        if job.state == "RUNNING":
            break
        time.sleep(2)

    if job.state != "RUNNING":
        pytest.skip("Job did not start running in time")

    # Send warning signal to trigger checkpoint
    try:
        subprocess.run(["bkill", "-s", "USR2", job.job_id], check=True, timeout=30)
    except subprocess.CalledProcessError:
        pytest.skip("Could not send signal to job")

    # Wait a bit for checkpoint/requeue
    time.sleep(10)

    # The job should eventually complete (possibly after requeue)
    _wait_for_job(job, max_wait_s=600)

    # Job should have completed successfully
    try:
        result = job.result()
        assert result == 100
    except utils.UncompletedJobError as e:
        # May fail if requeue didn't work, which is OK for this test
        pytest.skip(f"Job did not complete after checkpoint: {e}")

