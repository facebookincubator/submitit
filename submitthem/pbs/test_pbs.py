# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
import os
import signal
import subprocess
import typing as tp
from pathlib import Path
from unittest.mock import patch

import pytest

import submitthem

from .. import helpers
from ..core import job_environment, submission, test_core, utils
from ..core.core import Job
from . import pbs


def _mock_log_files(job: Job[tp.Any], prints: str = "", errors: str = "") -> None:
    """Write fake log files"""
    filepaths = [str(x).replace("%j", str(job.job_id)) for x in [job.paths.stdout, job.paths.stderr]]
    for filepath, msg in zip(filepaths, (prints, errors)):
        with Path(filepath).open("w") as f:
            f.write(msg)


class MockedPBSSubprocess(test_core.MockedSubprocess):
    """PBS-specific mocked subprocess that handles PBS directives"""

    INFO_HEADER = "Job ID           S"
    INFO_TEMPLATES = ("{j:<15} {state}",)

    def set_job_state(self, job_id: str, state: str, array: int = 0) -> None:
        # qstat column format uses a single-letter state; keep the first char.
        # If state is empty or only whitespace, keep it empty (will map to UNKNOWN in read_info)
        state_short = state[0] if state and state.strip() else ""
        super().set_job_state(job_id, state_short, array)

    def qstat(self, _: tp.Sequence[str]) -> str:
        return "\n".join(self.job_info.values())

    def qsub(self, args: tp.Sequence[str]) -> str:
        job_id = str(self.job_count)
        self.job_count += 1
        qsub_file = Path(args[0])
        array = self._get_array_count(qsub_file)
        self.set_job_state(job_id, "RUNNING", array)
        return f"Submitted batch job {job_id}\n"

    def qdel(self, _: tp.Sequence[str]) -> str:
        return ""

    def _get_array_count(self, submission_file: Path) -> int:
        """Parse PBS array specification from -J directive.
        Format: #PBS -J 0-12%3 or #PBS -J 0-12:3
        Returns the number of array tasks.
        """
        if not submission_file.exists():
            return 0
        array_lines = [line for line in submission_file.read_text().splitlines() if "-J" in line]
        if array_lines:
            # line should look like: #PBS -J 0-12%3 or #PBS -J 0-12:3
            # Extract the end index (12 in "0-12")
            array_str = array_lines[0].split(" 0-")[-1].split("%")[0].split(":")[0]
            return int(array_str) + 1
        return 0

    def get_job_context_env_vars(self, job_id: str) -> tp.Dict[str, str]:
        """Return PBS-specific environment variables for job execution"""
        return {
            "_USELESS_TEST_ENV_VAR_": "1",
            "SUBMITTHEM_EXECUTOR": "pbs",
            "PBS_JOBID": str(job_id),
        }



@contextlib.contextmanager
def mocked_pbs() -> tp.Iterator[MockedPBSSubprocess]:
    # TODO: check if both are needed
    mock = MockedPBSSubprocess(known_cmds=["qsub", "qsub -I"])
    try:
        with mock.context():
            yield mock
    finally:
        # Clear the state of the shared watcher
        pbs.PBSJob.watcher.clear()


def test_mocked_missing_state(tmp_path: Path) -> None:
    with mocked_pbs() as mock:
        mock.set_job_state("12", "       ")
        job: pbs.PBSJob[None] = pbs.PBSJob(tmp_path, "12")
        assert job.state == "UNKNOWN"
        job._interrupt(timeout=False)  # check_call is bypassed by MockedSubprocess


def test_job_environment() -> None:
    with mocked_pbs() as mock:
        mock.set_job_state("12", "RUNNING")
        with mock.job_context("12"):
            assert job_environment.JobEnvironment().cluster == "pbs"


def test_pbs_job_mocked(tmp_path: Path) -> None:
    with mocked_pbs() as mock:
        executor = pbs.PBSExecutor(folder=tmp_path)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
        # First mock job always have id 12
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
def test_pbs_job_array_mocked(use_batch_api: bool, tmp_path: Path) -> None:
    n = 5
    with mocked_pbs() as mock:
        executor = pbs.PBSExecutor(folder=tmp_path)
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
        assert [f"{array_id}_{a}" for a in range(n)] == [j.job_id for j in jobs]

        for job in jobs:
            assert job.state == "RUNNING"
            with mock.job_context(job.job_id):
                submission.process_job(job.paths.folder)
        # trying a pbs specific method
        jobs[0]._interrupt(timeout=True)  # type: ignore
        assert list(map(add, data1, data2)) == [j.result() for j in jobs]
        # check submission file
        qsub = Job(tmp_path, job_id=array_id).paths.submission_file.read_text()
        array_line = [line.strip() for line in qsub.splitlines() if "#PBS -J" in line]
        assert array_line == ["#PBS -J 0-4%3"]


def test_pbs_error_mocked(tmp_path: Path) -> None:
    with mocked_pbs() as mock:
        executor = pbs.PBSExecutor(folder=tmp_path)
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
    requeue = patch("submitthem.pbs.pbs.PBSJobEnvironment._requeue", return_value=None)
    with requeue as _patch:
        try:
            yield
        finally:
            if not_called:
                _patch.assert_not_called()
            else:
                _patch.assert_called_with(called_with)


def get_signal_handler(job: Job) -> job_environment.SignalHandler:
    env = pbs.PBSJobEnvironment()
    delayed = utils.DelayedSubmission.load(job.paths.submitted_pickle)
    sig = job_environment.SignalHandler(env, job.paths, delayed)
    return sig


def test_requeuing_checkpointable(tmp_path: Path, fast_forward_clock) -> None:
    usr_sig = submitthem.JobEnvironment._usr_sig()
    fs0 = helpers.FunctionSequence()
    fs0.add(test_core._three_time, 10)
    assert isinstance(fs0, helpers.Checkpointable)

    # Start job with a 60 minutes timeout
    with mocked_pbs():
        executor = pbs.PBSExecutor(folder=tmp_path, max_num_timeout=1)
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

    # This time the job as timed out,
    # but we have max_num_timeout=1, so we should requeue.
    # We are a little bit under the requested timedout, but close enough
    # to not consider this a preemption
    with pytest.raises(SystemExit), mock_requeue(called_with=0):
        sig.checkpoint_and_try_requeue(usr_sig)

    # Restart the job.
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=55)

    # The job has already timed out twice, we should stop here.
    usr_sig = pbs.PBSJobEnvironment._usr_sig()
    with mock_requeue(not_called=True), pytest.raises(
        utils.UncompletedJobError, match="timed-out too many times."
    ):
        sig.checkpoint_and_try_requeue(usr_sig)


def test_requeuing_not_checkpointable(tmp_path: Path, fast_forward_clock) -> None:
    usr_sig = submitthem.JobEnvironment._usr_sig()
    # Start job with a 60 minutes timeout
    with mocked_pbs():
        executor = pbs.PBSExecutor(folder=tmp_path, max_num_timeout=1)
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

    # Wait 50 minutes, now the job as timed out.
    with mock_requeue(not_called=True), pytest.raises(
        utils.UncompletedJobError, match="timed-out and not checkpointable"
    ):
        sig.checkpoint_and_try_requeue(usr_sig)


def test_checkpoint_and_exit(tmp_path: Path) -> None:
    usr_sig = submitthem.JobEnvironment._usr_sig()
    with mocked_pbs():
        executor = pbs.PBSExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(test_core._three_time, 10)

    sig = get_signal_handler(job)
    with pytest.raises(SystemExit), mock_requeue(not_called=True):
        sig.checkpoint_and_exit(usr_sig)

    # checkpoint_and_exit doesn't modify timeout counters.
    delayed = utils.DelayedSubmission.load(job.paths.submitted_pickle)
    assert delayed._timeout_countdown == 1


def test_make_qsub_string() -> None:
    string = pbs._make_qsub_string(
        command="my-custom_command",
        folder="/tmp",
        partition="learnfair",
        exclusive=True,
        additional_parameters={"blublu": 12},
    )
    # PBS uses -q for queue name, not partition
    assert "#PBS -q learnfair" in string
    assert "--command" not in string
    assert "constraint" not in string
    record_file = Path(__file__).parent / "_qsub_test_record.txt"
    if not record_file.exists():
        record_file.write_text(string)
    recorded = record_file.read_text()
    changes = []
    for k, (line1, line2) in enumerate(zip(string.splitlines(), recorded.splitlines())):
        if line1 != line2:
            changes.append(f'line #{k + 1}: "{line2}" -> "{line1}"')
    if changes:
        print(string)
        print("# # # # #")
        print(recorded)
        message = ["Difference with reference file:"] + changes
        message += ["", "Delete the record file if this is normal:", f"rm {record_file}"]
        raise AssertionError("\n".join(message))


def test_make_qsub_string_gpu() -> None:
    string = pbs._make_qsub_string(command="blublu", folder="/tmp", gpus_per_node=2)
    # PBS Pro uses -l select with ngpus, not --gpus-per-node
    assert "#PBS -l select=1:ncpus=1:ngpus=2" in string


def test_make_qsub_stderr() -> None:
    string = pbs._make_qsub_string(command="blublu", folder="/tmp", stderr_to_stdout=True)
    assert "#PBS -e" not in string


def test_update_parameters(tmp_path: Path) -> None:
    with mocked_pbs():
        executor = submitthem.AutoExecutor(folder=tmp_path)
    executor.update_parameters(mem_gb=3.5)
    assert executor._executor.parameters["mem"] == "3584MB"


def test_update_parameters_error(tmp_path: Path) -> None:
    with mocked_pbs():
        executor = pbs.PBSExecutor(folder=tmp_path)
    with pytest.raises(ValueError):
        executor.update_parameters(blublu=12)


def test_ntasks_per_node_conversion(tmp_path: Path) -> None:
    """Test that ntasks_per_node is properly converted to PBS mpiprocs format"""
    # Test: ntasks_per_node with nodes
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        nodes=2,
        ntasks_per_node=4,
    )
    # Should have mpiprocs=4 in select clause
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mpiprocs=4" in select_lines[0]
    assert "2:" in select_lines[0]  # 2 nodes


def test_ntasks_per_node_without_nodes(tmp_path: Path) -> None:
    """Test that ntasks_per_node works without explicit nodes parameter"""
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        ntasks_per_node=8,
    )
    # Should have select with mpiprocs
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mpiprocs=8" in select_lines[0]


def test_ntasks_per_node_with_cpus_per_task(tmp_path: Path) -> None:
    """Test that ntasks_per_node works with cpus_per_task"""
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        nodes=2,
        ntasks_per_node=4,
        cpus_per_task=2,
    )
    # Should have both mpiprocs and ncpus
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mpiprocs=4" in select_lines[0]
    assert "ncpus=2" in select_lines[0]


def test_gpus_per_task_conversion() -> None:
    """Test that gpus_per_task is properly converted to PBS ngpus format"""
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        gpus_per_task=2,
    )
    # Should have ngpus in select clause
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "ngpus=2" in select_lines[0]


def test_mem_per_gpu_conversion() -> None:
    """Test that mem_per_gpu is properly converted to PBS format"""
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        mem_per_gpu="4GB",
    )
    # Should have mem_per_gpu in select clause
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mem_per_gpu=4GB" in select_lines[0]


def test_mem_per_cpu_conversion() -> None:
    """Test that mem_per_cpu is properly converted to PBS format"""
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        mem_per_cpu="1GB",
    )
    # Should have mem_per_cpu in select clause
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mem_per_cpu=1GB" in select_lines[0]


def test_memory_string_parsing() -> None:
    """Test that memory values with units are properly parsed"""
    # Test with string format "512MB"
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        mem="512MB",
    )
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mem=512gb" in select_lines[0]

    # Test with numeric format
    string = pbs._make_qsub_string(
        command="test",
        folder="/tmp",
        mem="4",
    )
    select_lines = [line for line in string.splitlines() if "#PBS -l select" in line]
    assert len(select_lines) == 1
    assert "mem=4gb" in select_lines[0]


def test_num_gpus_deprecation() -> None:
    """Test that num_gpus parameter is deprecated"""
    with pytest.warns(DeprecationWarning, match="num_gpus.*deprecated"):
        string = pbs._make_qsub_string(
            command="test",
            folder="/tmp",
            num_gpus=2,
        )
    # Parameter should not appear in output since it was popped
    assert "num_gpus" not in string


def test_read_info() -> None:
    # Test parsing of qstat -f output with both simple jobs and array jobs
    example = """
Job Id: 5610980
    job_state = R
Job Id: 5610980.ext+
    job_state = R
Job Id: 5610980.0
    job_state = R
Job Id: 20956421_0
    job_state = R
Job Id: 20956421[2-4%25]
    job_state = Q
"""
    output = pbs.PBSInfoWatcher().read_info(example)
    print(f"{output=}")
    assert output["5610980"] == {"JobID": "5610980", "State": "RUNNING"}
    assert output["20956421_0"] == {"JobID": "20956421_0", "State": "RUNNING"}
    assert output["20956421_2"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert output["20956421_3"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert output["20956421_4"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert set(output) == {"5610980", "20956421_0", "20956421_2", "20956421_3", "20956421_4"}


def test_read_info_qstat_format() -> None:
    # qstat-like table format: Job ID ... State (state is the last column)
    example = """Job ID           S
5610980          R
5610980.ext+     R
5610980.0        R
20956421_0       R
20956421[2-4%25] Q
"""
    output = pbs.PBSInfoWatcher().read_info(example)
    assert output["5610980"] == {"JobID": "5610980", "State": "RUNNING"}
    assert output["20956421_0"] == {"JobID": "20956421_0", "State": "RUNNING"}
    assert output["20956421_2"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert output["20956421_3"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert output["20956421_4"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert set(output) == {"5610980", "20956421_0", "20956421_2", "20956421_3", "20956421_4"}


@pytest.mark.parametrize(  # type: ignore
    "name,state", [("12_0", "RUNNING"), ("12_1", "UNKNOWN"), ("12_2", "EXITING"), ("12_3", "UNKNOWN"), ("12_4", "EXITING")]
)
def test_read_info_array(name: str, state: str) -> None:
    example = "Job ID           S\n12_0             R\n12_[2,4-12]      X\n"
    watcher = pbs.PBSInfoWatcher()
    for jobid in ["12_2", "12_4"]:
        watcher.register_job(jobid)
    output = watcher.read_info(example)
    assert output.get(name, {}).get("State", "UNKNOWN") == state


@pytest.mark.parametrize(  # type: ignore
    "job_id,expected",
    [
        ("12", [(12,)]),
        ("12_0", [(12, 0)]),
        ("20_[2-7%56]", [(20, 2, 7)]),
        ("20_[2-7,12-17,22%56]", [(20, 2, 7), (20, 12, 17), (20, 22)]),
        ("20_[0%1]", [(20, 0)]),
    ],
)
def test_read_job_id(job_id: str, expected: tp.List[tp.Tuple[tp.Union[int, str], ...]]) -> None:
    output = pbs.read_job_id(job_id)
    assert output == [tuple(str(x) for x in group) for group in expected]


@pytest.mark.parametrize(  # type: ignore
    "string,expected",
    [(b"Submitted batch job 5610208\n", "5610208"), ("Submitted batch job 5610208\n", "5610208")],
)
def test_get_id_from_submission_command(string: str, expected: str) -> None:
    output = pbs.PBSExecutor._get_job_id_from_submission_command(string)
    assert output == expected


def test_get_id_from_submission_command_raise() -> None:
    with pytest.raises(utils.FailedSubmissionError):
        pbs.PBSExecutor._get_job_id_from_submission_command(string=b"blublu")


def test_watcher() -> None:
    with mocked_pbs() as mock:
        watcher = pbs.PBSInfoWatcher()
        mock.set_job_state("12", "RUNNING")
        assert watcher.num_calls == 0
        state = watcher.get_state(job_id="11")
        assert set(watcher._info_dict.keys()) == {"12"}
        assert watcher._registered == {"11"}

        assert state == "UNKNOWN"
        mock.set_job_state("12", "FAILED")
        state = watcher.get_state(job_id="12", mode="force")
        assert state == "FAILED"
        # TODO: this test is implementation specific. Not sure if we can rewrite it another way.
        assert watcher._registered == {"11", "12"}
        assert watcher._finished == {"12"}


def test_get_default_parameters() -> None:
    defaults = pbs._get_default_parameters()
    assert defaults["nodes"] == 1


def test_name() -> None:
    assert pbs.PBSExecutor.name() == "pbs"


@contextlib.contextmanager
def with_pbs_job_nodelist(node_list: str) -> tp.Iterator[pbs.PBSJobEnvironment]:
    os.environ["PBS_JOB_ID"] = "1"
    os.environ["PBS_JOB_NODELIST"] = node_list
    yield pbs.PBSJobEnvironment()
    del os.environ["PBS_JOB_NODELIST"]
    del os.environ["PBS_JOB_ID"]


def test_pbs_node_list() -> None:
    with with_pbs_job_nodelist("compute-b24") as env:
        assert ["compute-b24"] == env.hostnames
    with with_pbs_job_nodelist("compute-a1,compute-b2") as env:
        assert ["compute-a1", "compute-b2"] == env.hostnames
    with with_pbs_job_nodelist("compute-b2[1,2]") as env:
        assert ["compute-b21", "compute-b22"] == env.hostnames
    with with_pbs_job_nodelist("compute-b2[011,022]") as env:
        assert ["compute-b2011", "compute-b2022"] == env.hostnames
    with with_pbs_job_nodelist("compute-b2[1-3]") as env:
        assert ["compute-b21", "compute-b22", "compute-b23"] == env.hostnames
    with with_pbs_job_nodelist("compute-b2[1-3,5,6,8]") as env:
        assert [
            "compute-b21",
            "compute-b22",
            "compute-b23",
            "compute-b25",
            "compute-b26",
            "compute-b28",
        ] == env.hostnames
    with with_pbs_job_nodelist("compute-b2[1-3,5-6,8]") as env:
        assert [
            "compute-b21",
            "compute-b22",
            "compute-b23",
            "compute-b25",
            "compute-b26",
            "compute-b28",
        ] == env.hostnames
    with with_pbs_job_nodelist("compute-b2[1-3,5-6,8],compute-a1") as env:
        assert [
            "compute-b21",
            "compute-b22",
            "compute-b23",
            "compute-b25",
            "compute-b26",
            "compute-b28",
            "compute-a1",
        ] == env.hostnames
    with with_pbs_job_nodelist("compute[042,044]") as env:
        assert ["compute042", "compute044"] == env.hostnames
    with with_pbs_job_nodelist("compute[042-043,045,048-049]") as env:
        assert ["compute042", "compute043", "compute045", "compute048", "compute049"] == env.hostnames


def test_pbs_node_list_online_documentation() -> None:
    with with_pbs_job_nodelist("compute-b24-[1-3,5-9],compute-b25-[1,4,8]") as env:
        assert [
            "compute-b24-1",
            "compute-b24-2",
            "compute-b24-3",
            "compute-b24-5",
            "compute-b24-6",
            "compute-b24-7",
            "compute-b24-8",
            "compute-b24-9",
            "compute-b25-1",
            "compute-b25-4",
            "compute-b25-8",
        ] == env.hostnames


def test_pbs_invalid_parse() -> None:
    with pytest.raises(pbs.PBSParseException):
        with with_pbs_job_nodelist("compute-b2[1-,4]") as env:
            print(env.hostnames)
    with pytest.raises(pbs.PBSParseException):
        with with_pbs_job_nodelist("compute-b2[1,2,compute-b3]") as env:
            print(env.hostnames)


def test_pbs_missing_node_list() -> None:
    with with_pbs_job_nodelist("") as env:
        assert [env.hostname] == env.hostnames


def test_pbs_weird_dir(weird_tmp_path: Path) -> None:
    if "\n" in weird_tmp_path.name:
        pytest.skip("test doesn't support newline in 'weird_tmp_path'")
    with mocked_pbs():
        executor = pbs.PBSExecutor(folder=weird_tmp_path)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)

    # Touch the output files
    job.paths.stdout.write_text("")
    job.paths.stderr.write_text("")

    # Try to read PBS directives from the file
    qsub_args = {}
    for line in job.paths.submission_file.read_text().splitlines():
        if not line.startswith("#PBS"):
            continue
        # PBS format can be:
        # #PBS -o /path/to/file
        # #PBS -l select=...
        parts = line[len("#PBS"):].strip().split(None, 1)  # Split on first whitespace
        if len(parts) == 2:
            key, val = parts
            # Remove leading dash(es)
            key = key.lstrip('-')
            qsub_args[key] = val.replace("%j", job.job_id).replace("%t", "0")

    # Verify output and error files are properly quoted
    # PBS uses -o for output and -e for error
    if 'o' in qsub_args:
        subprocess.check_call("ls " + qsub_args["o"], shell=True)
    if 'e' in qsub_args:
        subprocess.check_call("ls " + qsub_args["e"], shell=True)


@pytest.mark.parametrize("params", [{}, {"mem_gb": None}])  # type: ignore
def test_pbs_through_auto(params: tp.Dict[str, int], tmp_path: Path) -> None:
    with mocked_pbs():
        executor = submitthem.AutoExecutor(folder=tmp_path)
        executor.update_parameters(**params, pbs_additional_parameters={"mem_per_gpu": 12})
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    # PBS uses -l select with mem specification, not --mem
    select_lines = [x for x in text.splitlines() if "#PBS -l select=" in x]
    # Check for any memory specification (mem=, mem_per_gpu=, or mem_per_cpu=)
    mem_lines = [x for x in text.splitlines() if "mem" in x and "#PBS" in x]
    assert len(select_lines) >= 1, f"No PBS select directive found. All lines:\n{text}"
    assert len(mem_lines) >= 1, f"No memory specification found. All lines:\n{text}"


def test_pbs_job_no_stderr(tmp_path: Path) -> None:
    def fail_silently():
        raise ValueError("Too bad")

    with mocked_pbs() as mock:
        executor = pbs.PBSExecutor(folder=tmp_path)
        # Failed but no stderr
        job = executor.submit(fail_silently)
        _mock_log_files(job, prints="job is running ...\n")
        job._results_timeout_s = 0
        with pytest.raises(utils.UncompletedJobError, match="job is running ..."):
            job._get_outcome_and_result()

        # Failed but no stderr nor stdout
        mock.set_job_state("13", "RUNNING")
        job = executor.submit(fail_silently)
        job._results_timeout_s = 0
        # Explicitly unlink stdout because submitthem is writing there on startup
        # job.paths.stdout.unlink()
        with pytest.raises(utils.UncompletedJobError, match="No output/error stream produced !"):
            job._get_outcome_and_result()
