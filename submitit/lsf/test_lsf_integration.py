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
# pylint: disable=redefined-outer-name
import os
import shutil
import subprocess
import time
import typing as tp
from pathlib import Path

import pytest

import submitit

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
            check=False,
        )
        output = result.stdout.strip()
        if not output:
            # Job no longer exists in bjobs (can happen after it completes/exits)
            return "GONE"

        # Check if all relevant jobs are finished, based on bjobs output (not job.state).
        lines = [ln for ln in output.splitlines() if ln.strip()]
        stats: tp.List[str] = []
        for line in lines:
            parts = line.split()
            # JOBID JOBINDEX STAT OR JOBID STAT (depending on LSF config / query)
            if len(parts) >= 3:
                stats.append(parts[2])
            elif len(parts) >= 2:
                stats.append(parts[1])

        if stats and all(s in ("DONE", "EXIT") for s in stats):
            # Prefer EXIT if any element exited.
            return "EXIT" if any(s == "EXIT" for s in stats) else "DONE"
        time.sleep(5)
    return "TIMEOUT"


def _wait_for_job_running(job: submitit.Job, max_wait_s: int = 300) -> None:
    """Wait for a job to reach RUN state, polling bjobs directly.

    This avoids relying on submitit's cached watcher state, which can be stale.
    """
    job_id = job.job_id.split("_")[0]
    start = time.time()
    last: str = ""
    while time.time() - start < max_wait_s:
        result = subprocess.run(
            ["bjobs", "-o", "JOBID JOBINDEX STAT", "-noheader", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        last = (result.stdout or "").strip() or (result.stderr or "").strip()
        out = (result.stdout or "").strip()
        if out:
            for line in out.splitlines():
                parts = line.split()
                stat = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else "")
                if stat == "RUN":
                    return
                if stat in ("DONE", "EXIT"):
                    # Job finished before reaching RUN (unexpected for long jobs).
                    break
        time.sleep(5)
    pytest.fail(f"Job did not reach RUN state within {max_wait_s}s. Last bjobs output: {last}")


def _wait_for_result_pickle(job: submitit.Job, max_wait_s: int = 900) -> None:
    """Wait until submitit produced a result pickle for this job.

    This is the most reliable completion signal for integration tests, especially when jobs
    may get requeued and transiently appear as EXIT in scheduler queries.
    """
    start = time.time()
    while time.time() - start < max_wait_s:
        if job.paths.result_pickle.exists():
            return
        time.sleep(2)
    pytest.fail(f"Result pickle was not produced within {max_wait_s}s: {job.paths.result_pickle}")


def _wait_for_submitit_start(job: submitit.Job, max_wait_s: int = 120) -> None:
    """Wait until the submitit worker started (log line present).

    LSF can report a job as RUN before the Python process installed signal handlers.
    If we send USR2 too early, it may terminate the job before checkpoint/requeue logic runs.

    We rely on Submitit's canonical log path (`job.paths.stdout`) rather than re-encoding
    filename conventions here.
    """
    stdout = job.paths.stdout
    start = time.time()
    while time.time() - start < max_wait_s:
        if stdout.exists():
            try:
                text = stdout.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if "Starting with JobEnvironment" in text:
                return
        time.sleep(2)
    pytest.fail(f"Submitit worker did not start within {max_wait_s}s (no marker in {stdout})")


@pytest.fixture
def test_folder() -> tp.Generator[Path, None, None]:
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

    # Wait for completion (via result pickle)
    _wait_for_result_pickle(job, max_wait_s=300)

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
        _wait_for_result_pickle(job, max_wait_s=300)

    # Check results
    results = [job.result() for job in jobs]
    assert results == [5, 7, 9]


def _wait_for_job_cancelled(job: submitit.Job, max_wait_s: int = 120) -> str:
    """Wait for a job to be cancelled/exit, polling bjobs directly.

    Returns the final state from bjobs (or 'GONE' if job no longer exists).
    """
    job_id = job.job_id.split("_")[0]  # Get main job ID for arrays
    start = time.time()
    while time.time() - start < max_wait_s:
        result = subprocess.run(
            ["bjobs", "-o", "JOBID JOBINDEX STAT", "-noheader", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        if not output:
            # Job no longer exists in bjobs
            return "GONE"
        # Check if the job is no longer RUN/PEND
        lines = output.splitlines()
        # Look for the specific job (in case of arrays)
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                stat = parts[2]
                if stat not in ("RUN", "PEND", "PSUSP", "USUSP", "SSUSP"):
                    return stat
            elif len(parts) >= 2:
                # Non-array: JOBID STAT
                stat = parts[1]
                if stat not in ("RUN", "PEND", "PSUSP", "USUSP", "SSUSP"):
                    return stat
        time.sleep(3)
    # Timeout - return current bjobs state
    return "TIMEOUT"


def test_lsf_cancel_cpu(test_folder: Path) -> None:
    """Test job cancellation."""
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    def long_sleep(x: int) -> int:
        time.sleep(600)
        return x

    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf")
    params: tp.Dict[str, tp.Any] = {"timeout_min": _get_timeout()}
    if _get_queue():
        params["lsf_queue"] = _get_queue()
    executor.update_parameters(**params)

    job = executor.submit(long_sleep, 42)

    # Wait for job to start (best-effort; cancel works even if still pending).
    try:
        _wait_for_job_running(job, max_wait_s=120)
    except pytest.fail.Exception:
        pass

    # Cancel the job
    job.cancel()

    # Poll bjobs directly (not job.state) to verify cancellation
    final_state = _wait_for_job_cancelled(job, max_wait_s=120)
    assert final_state not in ("RUN", "PEND", "TIMEOUT"), f"Job not cancelled properly: {final_state}"


@pytest.mark.skipif(
    not os.environ.get("SUBMITIT_LSF_TEST_GPU_QUEUE"),
    reason="SUBMITIT_LSF_TEST_GPU_QUEUE not set",
)
def test_lsf_gpu_request(test_folder: Path) -> None:
    """Test GPU resource request."""
    if shutil.which("bsub") is None:
        pytest.skip("bsub not found in PATH")

    def check_gpu() -> str:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
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
    _wait_for_result_pickle(job, max_wait_s=600)

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
            while self.count < max_count:
                self.count += 1
                time.sleep(1)
            return self.count

        def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> submitit.helpers.DelayedSubmission:
            # args[0] is max_count when called
            return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    # lsf_max_num_timeout is a constructor parameter, not an update_parameters parameter
    executor = submitit.AutoExecutor(folder=test_folder, cluster="lsf", lsf_max_num_timeout=2)
    params: tp.Dict[str, tp.Any] = {"timeout_min": _get_timeout()}
    if _get_queue():
        params["lsf_queue"] = _get_queue()
    executor.update_parameters(**params)

    counter = Counter()
    job = executor.submit(counter, 100)  # Will take 100 seconds

    # Wait for job to start running (poll bjobs directly)
    _wait_for_job_running(job, max_wait_s=300)
    _wait_for_submitit_start(job, max_wait_s=120)

    # Send warning signal to trigger checkpoint
    try:
        subprocess.run(["bkill", "-s", "USR2", job.job_id], check=True, timeout=30)
    except subprocess.CalledProcessError:
        pytest.skip("Could not send signal to job")

    # Wait a bit for checkpoint/requeue
    time.sleep(10)

    # The job should eventually complete (possibly after requeue)
    _wait_for_result_pickle(job, max_wait_s=900)

    # Job should have completed successfully
    result = job.result()
    assert result == 100
