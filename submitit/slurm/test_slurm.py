# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
import os
import signal
import subprocess
import tempfile
import typing as tp
from pathlib import Path
from unittest.mock import patch

import pytest

import submitit

from .. import helpers
from ..core import job_environment, submission, test_core, utils
from ..core.core import Job
from . import slurm


def _mock_log_files(job: Job[tp.Any], prints: str = "", errors: str = "") -> None:
    """Write fake log files"""
    filepaths = [str(x).replace("%j", str(job.job_id)) for x in [job.paths.stdout, job.paths.stderr]]
    for filepath, msg in zip(filepaths, (prints, errors)):
        with Path(filepath).open("w") as f:
            f.write(msg)


@contextlib.contextmanager
def mocked_slurm(state: str = "RUNNING", job_id: str = "12", array: int = 0) -> tp.Iterator[str]:
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            test_core.MockedSubprocess(state=state, job_id=job_id, shutil_which="srun", array=array).context()
        )
        envs = dict(_USELESS_TEST_ENV_VAR_="1", SUBMITIT_EXECUTOR="slurm", SLURM_JOB_ID=str(job_id))
        stack.enter_context(utils.environment_variables(**envs))
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        yield tmp


def test_mocked_missing_state() -> None:
    with mocked_slurm(state="       ", job_id="12") as tmp:
        job: slurm.SlurmJob[None] = slurm.SlurmJob(tmp, "12")
        assert job.state == "UNKNOWN"
        job._interrupt(timeout=False)  # check_call is bypassed by MockedSubprocess


def test_job_environment() -> None:
    with mocked_slurm(job_id="12"):
        assert job_environment.JobEnvironment().cluster == "slurm"


def test_slurm_job_mocked() -> None:
    with mocked_slurm() as tmp:
        executor = slurm.SlurmExecutor(folder=tmp)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
        assert job.job_id == "12"
        assert job.state == "RUNNING"
        assert job.stdout() is None
        _mock_log_files(job, errors="This is the error log\n", prints="hop")
        job._results_timeout_s = 0
        with pytest.raises(utils.UncompletedJobError):
            job._get_outcome_and_result()
        _mock_log_files(job, errors="This is the error log\n", prints="hop")
        submission.process_job(job.paths.folder)
        assert job.result() == 12
        # logs
        assert job.stdout() == "hop"
        assert job.stderr() == "This is the error log\n"
    assert "_USELESS_TEST_ENV_VAR_" not in os.environ, "Test context manager seems to be failing"


@pytest.mark.parametrize("context", (True, False))  # type: ignore
def test_slurm_job_array_mocked(context: bool) -> None:
    n = 5
    with mocked_slurm(array=n) as tmp:
        executor = slurm.SlurmExecutor(folder=tmp)
        executor.update_parameters(array_parallelism=3)
        data1, data2 = range(n), range(10, 10 + n)

        def add(x: int, y: int) -> int:
            assert x in data1
            assert y in data2
            return x + y

        jobs: tp.List[Job[int]] = []
        if not context:
            jobs = executor.map_array(add, data1, data2)
        else:
            with executor.batch():
                for d1, d2 in zip(data1, data2):
                    jobs.append(executor.submit(add, d1, d2))
        array_id = jobs[0].job_id.split("_")[0]
        assert [f"{array_id}_{a}" for a in range(n)] == [j.job_id for j in jobs]

        for job in jobs:
            os.environ["SLURM_JOB_ID"] = str(job.job_id)
            submission.process_job(job.paths.folder)
        # trying a slurm specific method
        jobs[0]._interrupt(timeout=True)  # type: ignore
        assert list(map(add, data1, data2)) == [j.result() for j in jobs]
        # check submission file
        sbatch = Job(tmp, job_id=array_id).paths.submission_file.read_text()
        array_line = [l.strip() for l in sbatch.splitlines() if "array" in l]
        assert array_line == ["#SBATCH --array=0-4%3"]


def test_slurm_error_mocked() -> None:
    with mocked_slurm() as tmp:
        executor = slurm.SlurmExecutor(folder=tmp)
        executor.update_parameters(time=24, gpus_per_node=0)  # just to cover the function
        job = executor.submit(test_core.do_nothing, 1, 2, error=12)
        with pytest.raises(ValueError):
            submission.process_job(job.paths.folder)
        _mock_log_files(job, errors="This is the error log\n")
        with pytest.raises(utils.FailedJobError):
            job.result()
        exception = job.exception()
        assert isinstance(exception, utils.FailedJobError)


@contextlib.contextmanager
def mock_requeue(called_with: int = None, not_called: bool = False):
    assert not_called or called_with is not None
    requeue = patch("submitit.slurm.slurm.SlurmJobEnvironment._requeue", return_value=None)
    try:
        with requeue as _patch:
            yield
    finally:
        if not_called:
            _patch.assert_not_called()
        else:
            _patch.assert_called_with(called_with)


def get_signal_handler(job: Job) -> job_environment.SignalHandler:
    env = slurm.SlurmJobEnvironment()
    delayed = utils.DelayedSubmission.load(job.paths.submitted_pickle)
    sig = job_environment.SignalHandler(env, job.paths, delayed)
    return sig


def test_requeuing_checkpointable(tmp_path: Path, fast_forward_clock) -> None:
    fs0 = helpers.FunctionSequence()
    fs0.add(test_core._three_time, 10)
    assert isinstance(fs0, helpers.Checkpointable)

    # Start job with a 60 minutes timeout
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(fs0)

    sig = get_signal_handler(job)

    fast_forward_clock(minutes=30)
    # Preempt the job after 30 minutes, the job hasn't timeout.
    with pytest.raises(SystemExit), mock_requeue(called_with=1):
        sig.checkpoint_and_try_requeue(signal.Signals.SIGUSR1)

    # Restart the job.
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=50)

    # This time the job as timed out,
    # but we have max_num_timeout=1, so we should requeue.
    # We are a little bit under the requested timedout, but close enough
    # to not consider this a preemption
    with pytest.raises(SystemExit), mock_requeue(called_with=0):
        sig.checkpoint_and_try_requeue(signal.Signals.SIGUSR1)

    # Restart the job.
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=55)

    # The job has already timed out twice, we should stop here.
    with mock_requeue(not_called=True), pytest.raises(
        utils.UncompletedJobError, match="timed-out too many times."
    ):
        sig.checkpoint_and_try_requeue(signal.Signals.SIGUSR1)


def test_requeuing_not_checkpointable(tmp_path: Path, fast_forward_clock) -> None:
    # Start job with a 60 minutes timeout
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(test_core._three_time, 10)

    # simulate job start
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=30)

    with mock_requeue(not_called=True):
        sig.bypass(signal.Signals.SIGTERM)

    # Preempt the job after 30 minutes, the job hasn't timeout.
    with pytest.raises(SystemExit), mock_requeue(called_with=1):
        sig.checkpoint_and_try_requeue(signal.Signals.SIGUSR1)

    # Restart the job from scratch
    sig = get_signal_handler(job)
    fast_forward_clock(minutes=50)

    # Wait 50 minutes, now the job as timed out.
    with mock_requeue(not_called=True), pytest.raises(
        utils.UncompletedJobError, match="timed-out and not checkpointable"
    ):
        sig.checkpoint_and_try_requeue(signal.Signals.SIGUSR1)


def test_checkpoint_and_exit(tmp_path: Path) -> None:
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(test_core._three_time, 10)

    sig = get_signal_handler(job)
    with pytest.raises(SystemExit), mock_requeue(not_called=True):
        sig.checkpoint_and_exit(signal.Signals.SIGUSR1)

    # checkpoint_and_exit doesn't modify timeout counters.
    delayed = utils.DelayedSubmission.load(job.paths.submitted_pickle)
    assert delayed._timeout_countdown == 1


def test_make_batch_string() -> None:
    string = slurm._make_sbatch_string(
        command="blublu",
        folder="/tmp",
        partition="learnfair",
        exclusive=True,
        additional_parameters=dict(blublu=12),
    )
    assert "partition" in string
    assert "--command" not in string
    assert "constraint" not in string
    record_file = Path(__file__).parent / "_sbatch_test_record.txt"
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


def test_make_batch_string_gpu() -> None:
    string = slurm._make_sbatch_string(command="blublu", folder="/tmp", gpus_per_node=2)
    assert "--gpus-per-node=2" in string


def test_make_batch_stderr() -> None:
    string = slurm._make_sbatch_string(command="blublu", folder="/tmp", stderr_to_stdout=True)
    assert "--error" not in string


def test_update_parameters_error() -> None:
    with mocked_slurm() as tmp:
        with pytest.raises(ValueError):
            executor = slurm.SlurmExecutor(folder=tmp)
            executor.update_parameters(blublu=12)


def test_read_info() -> None:
    example = """JobID|State
5610980|RUNNING
5610980.ext+|RUNNING
5610980.0|RUNING
20956421_0|RUNNING
20956421_[2-4%25]|PENDING
"""
    output = slurm.SlurmInfoWatcher().read_info(example)
    assert output["5610980"] == {"JobID": "5610980", "State": "RUNNING"}
    assert output["20956421_2"] == {"JobID": "20956421_[2-4%25]", "State": "PENDING"}
    assert set(output) == {"5610980", "20956421_0", "20956421_2", "20956421_3", "20956421_4"}


@pytest.mark.parametrize(  # type: ignore
    "name,state", [("12_0", "R"), ("12_1", "U"), ("12_2", "X"), ("12_3", "U"), ("12_4", "X")]
)
def test_read_info_array(name: str, state: str) -> None:
    example = "JobID|State\n12_0|R\n12_[2,4-12]|X"
    watcher = slurm.SlurmInfoWatcher()
    for jobid in ["12_2", "12_4"]:
        watcher.register_job(jobid)
    output = watcher.read_info(example)
    assert output.get(name, {}).get("State", "U") == state


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
    output = slurm.read_job_id(job_id)
    assert output == [tuple(str(x) for x in group) for group in expected]


@pytest.mark.parametrize(  # type: ignore
    "string,expected",
    [(b"Submitted batch job 5610208\n", "5610208"), ("Submitted batch job 5610208\n", "5610208")],
)
def test_get_id_from_submission_command(string: str, expected: str) -> None:
    output = slurm.SlurmExecutor._get_job_id_from_submission_command(string)
    assert output == expected


def test_get_id_from_submission_command_raise() -> None:
    with pytest.raises(utils.FailedSubmissionError):
        slurm.SlurmExecutor._get_job_id_from_submission_command(string=b"blublu")


def test_watcher() -> None:
    with mocked_slurm():
        watcher = slurm.SlurmInfoWatcher()
        assert watcher.num_calls == 0
        state = watcher.get_state(job_id="11")
        assert state == "UNKNOWN"
        assert set(watcher._info_dict.keys()) == {"12"}
        watcher.clear()
        assert watcher._registered == {"11"}


def test_get_default_parameters() -> None:
    defaults = slurm._get_default_parameters()
    assert defaults["nodes"] == 1


def test_name() -> None:
    assert slurm.SlurmExecutor.name() == "slurm"


@contextlib.contextmanager
def with_slurm_job_nodelist(node_list: str) -> tp.Iterator[slurm.SlurmJobEnvironment]:
    os.environ["SLURM_JOB_ID"] = "1"
    os.environ["SLURM_JOB_NODELIST"] = node_list
    yield slurm.SlurmJobEnvironment()
    del os.environ["SLURM_JOB_NODELIST"]
    del os.environ["SLURM_JOB_ID"]


def test_slurm_node_list() -> None:
    with with_slurm_job_nodelist("compute-b24") as env:
        assert ["compute-b24"] == env.hostnames
    with with_slurm_job_nodelist("compute-a1,compute-b2") as env:
        assert ["compute-a1", "compute-b2"] == env.hostnames
    with with_slurm_job_nodelist("compute-b2[1,2]") as env:
        assert ["compute-b21", "compute-b22"] == env.hostnames
    with with_slurm_job_nodelist("compute-b2[011,022]") as env:
        assert ["compute-b2011", "compute-b2022"] == env.hostnames
    with with_slurm_job_nodelist("compute-b2[1-3]") as env:
        assert ["compute-b21", "compute-b22", "compute-b23"] == env.hostnames
    with with_slurm_job_nodelist("compute-b2[1-3,5,6,8]") as env:
        assert [
            "compute-b21",
            "compute-b22",
            "compute-b23",
            "compute-b25",
            "compute-b26",
            "compute-b28",
        ] == env.hostnames
    with with_slurm_job_nodelist("compute-b2[1-3,5-6,8]") as env:
        assert [
            "compute-b21",
            "compute-b22",
            "compute-b23",
            "compute-b25",
            "compute-b26",
            "compute-b28",
        ] == env.hostnames
    with with_slurm_job_nodelist("compute-b2[1-3,5-6,8],compute-a1") as env:
        assert [
            "compute-b21",
            "compute-b22",
            "compute-b23",
            "compute-b25",
            "compute-b26",
            "compute-b28",
            "compute-a1",
        ] == env.hostnames
    with with_slurm_job_nodelist("compute[042,044]") as env:
        assert ["compute042", "compute044"] == env.hostnames
    with with_slurm_job_nodelist("compute[042-043,045,048-049]") as env:
        assert ["compute042", "compute043", "compute045", "compute048", "compute049"] == env.hostnames


def test_slurm_node_list_online_documentation() -> None:
    with with_slurm_job_nodelist("compute-b24-[1-3,5-9],compute-b25-[1,4,8]") as env:
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


def test_slurm_invalid_parse() -> None:
    with pytest.raises(slurm.SlurmParseException):
        with with_slurm_job_nodelist("compute-b2[1-,4]") as env:
            print(env.hostnames)
    with pytest.raises(slurm.SlurmParseException):
        with with_slurm_job_nodelist("compute-b2[1,2,compute-b3]") as env:
            print(env.hostnames)


def test_slurm_missing_node_list() -> None:
    with with_slurm_job_nodelist("") as env:
        assert [env.hostname] == env.hostnames


def test_slurm_weird_dir(weird_tmp_path: Path) -> None:
    if "\n" in weird_tmp_path.name:
        pytest.skip("test doesn't support newline in 'weird_tmp_path'")
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=weird_tmp_path)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)

    # Touch the ouputfiles
    job.paths.stdout.write_text("")
    job.paths.stderr.write_text("")

    # Try to read sbatch flags from the file like sbatch would do it.
    sbatch_args = {}
    for l in job.paths.submission_file.read_text().splitlines():
        if not l.startswith("#SBATCH"):
            continue
        if "=" not in l:
            continue
        key, val = l[len("#SBATCH") :].strip().split("=", 1)
        sbatch_args[key] = val.replace("%j", job.job_id).replace("%t", "0")

    # We do not quote --output and --error values here,
    # because we want to check if they have been properly quoted before.
    subprocess.check_call("ls " + sbatch_args["--output"], shell=True)
    subprocess.check_call("ls " + sbatch_args["--error"], shell=True)


@pytest.mark.parametrize("params", [{}, {"mem_gb": None}])  # type: ignore
def test_slurm_through_auto(params: tp.Dict[str, int], tmp_path: Path) -> None:
    with mocked_slurm():
        executor = submitit.AutoExecutor(folder=tmp_path)
        executor.update_parameters(**params, slurm_additional_parameters={"mem_per_gpu": 12})
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    mem_lines = [x for x in text.splitlines() if "#SBATCH --mem" in x]
    assert len(mem_lines) == 1, f"Unexpected lines: {mem_lines}"


def test_slurm_job_no_stderr() -> None:
    def fail_silently():
        raise ValueError("Too bad")

    with mocked_slurm() as tmp:
        executor = slurm.SlurmExecutor(folder=tmp)
        # Failed but no stderr
        job = executor.submit(fail_silently)
        _mock_log_files(job, prints="job is running ...\n")
        job._results_timeout_s = 0
        with pytest.raises(utils.UncompletedJobError, match="job is running ..."):
            job._get_outcome_and_result()

        # Failed but no stderr nor stdout
        job = executor.submit(fail_silently)
        job._results_timeout_s = 0
        # Explicitly unlink stdout because submitit is writing there on startup
        job.paths.stdout.unlink()
        with pytest.raises(utils.UncompletedJobError, match="No output/error stream produced !"):
            job._get_outcome_and_result()
