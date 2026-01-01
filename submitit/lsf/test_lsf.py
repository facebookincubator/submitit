# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
import os
import signal
import typing as tp
from pathlib import Path
from unittest.mock import patch

import pytest

import submitit

from ..core import job_environment, submission, test_core, utils
from ..core.core import Job
from . import lsf


def _mock_log_files(job: Job[tp.Any], prints: str = "", errors: str = "") -> None:
    """Write fake log files"""
    filepaths = [str(x).replace("%j", str(job.job_id)) for x in [job.paths.stdout, job.paths.stderr]]
    for filepath, msg in zip(filepaths, (prints, errors)):
        with Path(filepath).open("w") as f:
            f.write(msg)


class MockedLsfSubprocess:
    """Helper for mocking LSF subprocess calls.

    Mimics real LSF bjobs output format: "JOBID JOBINDEX STAT"
    - Non-array jobs: "12345 0 RUN"
    - Array jobs: "12345 1 RUN" (1-based indexing)
    """

    def __init__(self, known_cmds: tp.Optional[tp.Sequence[str]] = None) -> None:
        # job_bjobs stores: {(job_id, job_index): state}
        # job_index=0 for non-array, 1+ for array elements (1-based)
        self.job_bjobs: tp.Dict[tp.Tuple[str, int], str] = {}
        self.last_job: str = ""
        self._subprocess_check_output = __import__("subprocess").check_output
        self.known_cmds = known_cmds or []
        self.job_count = 12

    def __call__(self, command: tp.Sequence[str], **kwargs: tp.Any) -> bytes:
        program = command[0]
        if program == "bjobs":
            return self.bjobs(command[1:]).encode()
        elif program == "bsub":
            # Handle stdin-based submission
            stdin = kwargs.get("stdin")
            if stdin:
                return self.bsub_stdin(stdin).encode()
            return self.bsub(command[1:]).encode()
        elif program in ["bkill", "brequeue"]:
            return b""
        elif program == "tail":
            return self._subprocess_check_output(command, **kwargs)
        else:
            raise ValueError(f'Unknown command to mock "{command}".')

    def bjobs(self, args: tp.Sequence[str]) -> str:
        # Return status for all tracked jobs in format: "JOBID JOBINDEX STAT"
        lines = []
        for (job_id, job_index), state in self.job_bjobs.items():
            lines.append(f"{job_id} {job_index} {state}")
        return "\n".join(lines)

    def bsub(self, args: tp.Sequence[str]) -> str:
        """Create a "RUN" job from command line args."""
        return self._create_job()

    def bsub_stdin(self, stdin_file: tp.IO[str]) -> str:
        """Create a "RUN" job from stdin script."""
        content = stdin_file.read()
        # Check for array job - LSF uses 1-based indexing: [1-N]
        array_match = None
        for line in content.splitlines():
            if "#BSUB -J" in line and "[" in line:
                import re

                array_match = re.search(r"\[(\d+)-(\d+)\]", line)
                break

        if array_match:
            start = int(array_match.group(1))
            end = int(array_match.group(2))
            array_size = end - start + 1
            return self._create_job(array_size=array_size, start_index=start)
        return self._create_job()

    def _create_job(self, array_size: int = 0, start_index: int = 1) -> str:
        job_id = str(self.job_count)
        self.job_count += 1
        if array_size > 0:
            # Array job - create entries for each element with 1-based indexing
            for i in range(array_size):
                self.set_job_state(f"{job_id}", "RUN", job_index=start_index + i)
        else:
            # Non-array job has JOBINDEX=0
            self.set_job_state(job_id, "RUN", job_index=0)
        return f"Job <{job_id}> is submitted to queue <normal>.\n"

    def set_job_state(self, job_id: str, state: str, job_index: int = 0) -> None:
        """Set job state.

        Args:
            job_id: The LSF job ID
            state: LSF state (RUN, PEND, DONE, EXIT, etc.)
            job_index: 0 for non-array jobs, 1+ for array elements (1-based)
        """
        self.job_bjobs[(job_id, job_index)] = state
        self.last_job = job_id

    def which(self, name: str) -> tp.Optional[str]:
        return "here" if name in self.known_cmds else None

    @contextlib.contextmanager
    def context(self) -> tp.Iterator[None]:
        with patch("subprocess.check_output", new=self):
            with patch("shutil.which", new=self.which):
                with patch("subprocess.check_call", new=lambda *args, **kwargs: None):
                    yield None

    @contextlib.contextmanager
    def job_context(self, job_id: str) -> tp.Iterator[None]:
        # Parse job_id for array jobs (format: 12_0 -> array job 12, index 0)
        env_vars = {
            "_USELESS_TEST_ENV_VAR_": "1",
            "SUBMITIT_EXECUTOR": "lsf",
            "LSB_JOBID": str(job_id.split("_")[0]),
            "SUBMITIT_LSF_NTASKS": "1",
            "SUBMITIT_LSF_NNODES": "1",
            "SUBMITIT_LSF_NODEID": "0",
            "SUBMITIT_LSF_GLOBAL_RANK": "0",
            "SUBMITIT_LSF_LOCAL_RANK": "0",
        }
        if "_" in job_id:
            main_id, array_idx = job_id.split("_", 1)
            env_vars["SUBMITIT_LSF_ARRAY_JOB_ID"] = main_id
            env_vars["SUBMITIT_LSF_ARRAY_TASK_ID"] = array_idx
        with utils.environment_variables(**env_vars):
            yield None


@contextlib.contextmanager
def mocked_lsf() -> tp.Iterator[MockedLsfSubprocess]:
    mock = MockedLsfSubprocess(known_cmds=["bsub", "bjobs", "bkill", "brequeue"])
    try:
        with mock.context():
            yield mock
    finally:
        # Clear the state of the shared watcher
        lsf.LsfJob.watcher.clear()


def test_mocked_missing_state(tmp_path: Path) -> None:
    with mocked_lsf() as mock:
        mock.set_job_state("12", "UNKWN")
        job: lsf.LsfJob[None] = lsf.LsfJob(tmp_path, "12")
        assert job.state == "UNKNOWN"


def test_job_environment() -> None:
    with mocked_lsf() as mock:
        mock.set_job_state("12", "RUN")
        with mock.job_context("12"):
            assert job_environment.JobEnvironment().cluster == "lsf"


def test_lsf_job_mocked(tmp_path: Path) -> None:
    with mocked_lsf() as mock:
        executor = lsf.LsfExecutor(folder=tmp_path)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
        # First mock job always has id 12
        assert job.job_id == "12"
        assert job.state == "RUNNING"
        assert job.stdout() is None
        _mock_log_files(job, errors="This is the error log\n", prints="hop")
        job._results_timeout_s = 0
        with pytest.raises(utils.UncompletedJobError):
            job._get_outcome_and_result()
        _mock_log_files(job, errors="This is the error log\n", prints="hop")

        with mock.job_context(job.job_id):
            submission.process_job(job.paths.folder)
        assert job.result() == 12
        # logs
        assert job.stdout() == "hop"
        assert job.stderr() == "This is the error log\n"
        assert "_USELESS_TEST_ENV_VAR_" not in os.environ, "Test context manager seems to be failing"


@pytest.mark.parametrize("use_batch_api", (False, True))  # type: ignore
def test_lsf_job_array_mocked(use_batch_api: bool, tmp_path: Path) -> None:
    n = 5
    with mocked_lsf() as mock:
        executor = lsf.LsfExecutor(folder=tmp_path)
        executor.update_parameters(array_parallelism=3)
        data1, data2 = range(n), range(10, 10 + n)

        def add(x: int, y: int) -> int:
            assert x in data1
            assert y in data2
            return x + y

        jobs: tp.List[Job[int]] = []
        if use_batch_api:
            with executor.batch():
                for d1, d2 in zip(data1, data2):
                    jobs.append(executor.submit(add, d1, d2))
        else:
            jobs = executor.map_array(add, data1, data2)
        array_id = jobs[0].job_id.split("_")[0]
        # LSF arrays are 1-based, so indices go from 1 to n
        assert [f"{array_id}_{a}" for a in range(1, n + 1)] == [j.job_id for j in jobs]

        for job in jobs:
            assert job.state == "RUNNING"
            with mock.job_context(job.job_id):
                submission.process_job(job.paths.folder)
        assert list(map(add, data1, data2)) == [j.result() for j in jobs]


def test_lsf_error_mocked(tmp_path: Path) -> None:
    with mocked_lsf() as mock:
        executor = lsf.LsfExecutor(folder=tmp_path)
        executor.update_parameters(time=24, gpus_per_node=0)  # just to cover the function
        job = executor.submit(test_core.do_nothing, 1, 2, error=12)
        with mock.job_context(job.job_id):
            with pytest.raises(ValueError):
                submission.process_job(job.paths.folder)
        _mock_log_files(job, errors="This is the error log\n")
        with pytest.raises(utils.FailedJobError):
            job.result()
        exception = job.exception()
        assert isinstance(exception, utils.FailedJobError)


@contextlib.contextmanager
def mock_requeue(called_with: tp.Optional[int] = None, not_called: bool = False):
    assert not_called or called_with is not None
    requeue = patch("submitit.lsf.lsf.LsfJobEnvironment._requeue", return_value=None)
    with requeue as _patch:
        try:
            yield
        finally:
            if not_called:
                _patch.assert_not_called()
            else:
                _patch.assert_called_with(called_with)


def get_signal_handler(job: Job) -> job_environment.SignalHandler:
    env = lsf.LsfJobEnvironment()
    delayed = utils.DelayedSubmission.load(job.paths.submitted_pickle)
    sig = job_environment.SignalHandler(env, job.paths, delayed)
    return sig


def test_requeuing_checkpointable(tmp_path: Path, fast_forward_clock) -> None:
    usr_sig = submitit.JobEnvironment._usr_sig()
    fs0 = submitit.helpers.FunctionSequence()
    fs0.add(test_core._three_time, 10)
    assert isinstance(fs0, submitit.helpers.Checkpointable)

    # Start job with a 60 minutes timeout
    with mocked_lsf():
        executor = lsf.LsfExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(fs0)

    sig = get_signal_handler(job)

    fast_forward_clock(minutes=30)
    # Preempt the job after 30 minutes, the job hasn't timeout.
    with pytest.raises(SystemExit), mock_requeue(called_with=1):
        sig.checkpoint_and_try_requeue(usr_sig)

    # Restart the job.
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=50)

    # This time the job has timed out,
    # but we have max_num_timeout=1, so we should requeue.
    with pytest.raises(SystemExit), mock_requeue(called_with=0):
        sig.checkpoint_and_try_requeue(usr_sig)

    # Restart the job.
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=55)

    # The job has already timed out twice, we should stop here.
    usr_sig = lsf.LsfJobEnvironment._usr_sig()
    with mock_requeue(not_called=True), pytest.raises(
        utils.UncompletedJobError, match="timed-out too many times."
    ):
        sig.checkpoint_and_try_requeue(usr_sig)


def test_requeuing_not_checkpointable(tmp_path: Path, fast_forward_clock) -> None:
    usr_sig = submitit.JobEnvironment._usr_sig()
    # Start job with a 60 minutes timeout
    with mocked_lsf():
        executor = lsf.LsfExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(test_core._three_time, 10)

    # simulate job start
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=30)

    with mock_requeue(not_called=True):
        sig.bypass(signal.Signals.SIGTERM)

    # Preempt the job after 30 minutes, the job hasn't timeout.
    with pytest.raises(SystemExit), mock_requeue(called_with=1):
        sig.checkpoint_and_try_requeue(usr_sig)

    # Restart the job from scratch
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=50)

    # Wait 50 minutes, now the job has timed out.
    with mock_requeue(not_called=True), pytest.raises(
        utils.UncompletedJobError, match="timed-out and not checkpointable"
    ):
        sig.checkpoint_and_try_requeue(usr_sig)


def test_make_bsub_string() -> None:
    string = lsf._make_bsub_string(
        command="blublu bar",
        folder="/tmp",
        queue="normal",
        additional_parameters={"x": "value"},
    )
    assert "#BSUB -q normal" in string
    assert "--command" not in string
    assert "blublu bar" in string


def test_make_bsub_string_gpu() -> None:
    string = lsf._make_bsub_string(command="blublu", folder="/tmp", gpus_per_node=2)
    assert 'num=2' in string


def test_make_bsub_stderr() -> None:
    string = lsf._make_bsub_string(command="blublu", folder="/tmp", stderr_to_stdout=True)
    assert "#BSUB -e" not in string


def test_update_parameters(tmp_path: Path) -> None:
    with mocked_lsf():
        executor = submitit.AutoExecutor(folder=tmp_path, cluster="lsf")
    executor.update_parameters(mem_gb=3.5)
    assert executor._executor.parameters["mem"] == "3584M"


def test_update_parameters_error(tmp_path: Path) -> None:
    with mocked_lsf():
        executor = lsf.LsfExecutor(folder=tmp_path)
    with pytest.raises(ValueError):
        executor.update_parameters(blublu=12)


def test_read_info() -> None:
    """Test parsing of bjobs -o 'JOBID JOBINDEX STAT' -noheader output.

    Real LSF output format:
    - Non-array job: "301828 0 DONE" (JOBINDEX=0 means non-array)
    - Array job element: "301970 1 RUN" (1-based array index)
    """
    example = """12345 0 RUN
12346 0 PEND
12347 1 RUN
12347 2 PEND
"""
    output = lsf.LsfInfoWatcher().read_info(example)
    assert output["12345"] == {"JobID": "12345", "State": "RUNNING"}
    assert output["12346"] == {"JobID": "12346", "State": "PENDING"}
    # Array elements use 1-based indexing in LSF
    assert output["12347_1"] == {"JobID": "12347[1]", "State": "RUNNING"}
    assert output["12347_2"] == {"JobID": "12347[2]", "State": "PENDING"}


def test_read_info_states() -> None:
    """Test all LSF state normalizations."""
    watcher = lsf.LsfInfoWatcher()
    assert watcher._normalize_state("PEND") == "PENDING"
    assert watcher._normalize_state("RUN") == "RUNNING"
    assert watcher._normalize_state("DONE") == "COMPLETED"
    assert watcher._normalize_state("EXIT") == "FAILED"
    assert watcher._normalize_state("PSUSP") == "SUSPENDED"
    assert watcher._normalize_state("USUSP") == "SUSPENDED"
    assert watcher._normalize_state("SSUSP") == "SUSPENDED"
    assert watcher._normalize_state("WAIT") == "PENDING"
    assert watcher._normalize_state("ZOMBI") == "FAILED"
    assert watcher._normalize_state("UNKNOWN_STATE") == "UNKNOWN"


@pytest.mark.parametrize(  # type: ignore
    "string,expected",
    [
        (b"Job <5610208> is submitted to queue <normal>.\n", "5610208"),
        ("Job <5610208> is submitted to queue <normal>.\n", "5610208"),
    ],
)
def test_get_id_from_submission_command(string: str, expected: str) -> None:
    output = lsf.LsfExecutor._get_job_id_from_submission_command(string)
    assert output == expected


def test_get_id_from_submission_command_raise() -> None:
    with pytest.raises(utils.FailedSubmissionError):
        lsf.LsfExecutor._get_job_id_from_submission_command(string=b"blublu")


def test_watcher() -> None:
    with mocked_lsf() as mock:
        watcher = lsf.LsfInfoWatcher()
        mock.set_job_state("12", "RUN")
        assert watcher.num_calls == 0
        state = watcher.get_state(job_id="11")
        assert watcher._registered == {"11"}

        assert state == "UNKNOWN"
        mock.set_job_state("12", "EXIT")
        state = watcher.get_state(job_id="12", mode="force")
        assert state == "FAILED"
        assert watcher._registered == {"11", "12"}
        assert watcher._finished == {"12"}


def test_get_default_parameters() -> None:
    defaults = lsf._get_default_parameters()
    assert defaults["nodes"] == 1


def test_name() -> None:
    assert lsf.LsfExecutor.name() == "lsf"


def test_lsf_job_environment_array() -> None:
    """Test that array job environment is correctly set up."""
    with mocked_lsf() as mock:
        # LSF arrays are 1-based, so first element has index 1
        mock.set_job_state("12", "RUN", job_index=1)
        with mock.job_context("12_1"):
            env = job_environment.JobEnvironment()
            assert env.cluster == "lsf"
            assert env.array_job_id == "12"
            assert env.array_task_id == "1"
            assert env.job_id == "12_1"


def test_lsf_job_environment_simple() -> None:
    """Test that simple job environment is correctly set up."""
    with mocked_lsf() as mock:
        mock.set_job_state("12", "RUN")
        with mock.job_context("12"):
            env = job_environment.JobEnvironment()
            assert env.cluster == "lsf"
            assert env.array_job_id is None
            assert env.array_task_id is None
            assert env.job_id == "12"


def test_read_job_id() -> None:
    """Test job ID parsing."""
    # Simple job
    assert lsf.read_job_id("12345") == [("12345",)]
    # Submitit internal format
    assert lsf.read_job_id("12345_0") == [("12345", "0")]
    # LSF array format
    assert lsf.read_job_id("12345[1-3]") == [("12345", "1", "3")]


def test_lsf_through_auto(tmp_path: Path) -> None:
    with mocked_lsf():
        executor = submitit.AutoExecutor(folder=tmp_path, cluster="lsf")
        executor.update_parameters(lsf_additional_parameters={"x": "value"})
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    assert "#BSUB" in text


# ============================================================
# Real LSF output fixtures - captured from actual LSF cluster
# ============================================================


def test_parse_real_bsub_output() -> None:
    """Test parsing of real bsub output captured from LSF cluster.

    Real outputs from IBM Spectrum LSF 10.1:
    - Includes memory reservation info before the job ID line
    - Job ID is in format: Job <ID> is submitted to ...
    """
    # Real bsub output (with extra info lines that should be ignored)
    real_output = """Memory reservation is (MB): 1024
Memory Limit is (MB): 1024

===Your total amount of memory reservation for this job is (MB): 1024 ===

Job <301828> is submitted to default queue <short>.
"""
    job_id = lsf.LsfExecutor._get_job_id_from_submission_command(real_output)
    assert job_id == "301828"


def test_parse_real_bjobs_output_single() -> None:
    """Test parsing of real bjobs output for a single (non-array) job.

    Format from: bjobs -o "JOBID JOBINDEX STAT" -noheader <jobid>
    For non-array jobs, JOBINDEX is always 0.
    """
    # Real bjobs output for a completed non-array job
    real_output = "301828 0 DONE\n"
    output = lsf.LsfInfoWatcher().read_info(real_output)
    assert output["301828"] == {"JobID": "301828", "State": "COMPLETED"}


def test_parse_real_bjobs_output_array() -> None:
    """Test parsing of real bjobs output for an array job.

    Format from: bjobs -o "JOBID JOBINDEX STAT" -noheader <jobid>
    For array jobs, JOBINDEX is 1-based (1, 2, 3, ...).
    """
    # Real bjobs output for a 3-element array job
    real_output = """301970 1 RUN
301970 2 RUN
301970 3 DONE
"""
    output = lsf.LsfInfoWatcher().read_info(real_output)

    # Main job ID should be stored
    assert "301970" in output

    # Array elements should be stored with 1-based indices
    assert output["301970_1"] == {"JobID": "301970[1]", "State": "RUNNING"}
    assert output["301970_2"] == {"JobID": "301970[2]", "State": "RUNNING"}
    assert output["301970_3"] == {"JobID": "301970[3]", "State": "COMPLETED"}


def test_parse_real_bjobs_output_mixed_states() -> None:
    """Test parsing bjobs output with various real LSF states."""
    # Simulated output with various states seen in real LSF
    real_output = """316561 0 PEND
316693 1 PEND
316693 2 RUN
316693 3 DONE
317063 0 EXIT
"""
    output = lsf.LsfInfoWatcher().read_info(real_output)

    assert output["316561"] == {"JobID": "316561", "State": "PENDING"}
    assert output["316693_1"] == {"JobID": "316693[1]", "State": "PENDING"}
    assert output["316693_2"] == {"JobID": "316693[2]", "State": "RUNNING"}
    assert output["316693_3"] == {"JobID": "316693[3]", "State": "COMPLETED"}
    assert output["317063"] == {"JobID": "317063", "State": "FAILED"}

