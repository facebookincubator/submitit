# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
import os
import shlex
import signal
import subprocess
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
def mocked_slurm() -> tp.Iterator[test_core.MockedSubprocess]:
    mock = test_core.MockedSubprocess(known_cmds=["srun"])
    try:
        with mock.context():
            yield mock
    finally:
        # Clear the state of the shared watcher
        slurm.SlurmJob.watcher.clear()


def test_mocked_missing_state(tmp_path: Path) -> None:
    with mocked_slurm() as mock:
        mock.set_job_state("12", "       ")
        job: slurm.SlurmJob[None] = slurm.SlurmJob(tmp_path, "12")
        assert job.state == "UNKNOWN"
        job._interrupt(timeout=False)  # check_call is bypassed by MockedSubprocess


def test_job_environment() -> None:
    with mocked_slurm() as mock:
        mock.set_job_state("12", "RUNNING")
        with mock.job_context("12"):
            assert job_environment.JobEnvironment().cluster == "slurm"


def test_slurm_job_mocked(tmp_path: Path) -> None:
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path)
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


def _run_submission_script(script: Path) -> None:
    command = next(line for line in script.read_text().splitlines() if "-m submitit.core._submit" in line)
    args = shlex.split(command)
    with patch("sys.argv", args[args.index("-m") + 1 :]):
        submission.submitit_main()


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("folder_pattern", ["", "%j", "%A"])
def test_slurm_submission_script_reuse(tmp_path: Path, count: int, folder_pattern: str) -> None:
    folder = tmp_path / "job's logs" / folder_pattern
    with mocked_slurm() as mock, patch("submitit.core.submission.time.sleep"):
        executor = slurm.SlurmExecutor(folder=folder)
        jobs = executor.map_array(test_core._three_time, range(1, count + 1))
        script = jobs[0].paths.submission_file
        for job in jobs:
            with mock.job_context(job.job_id):
                _run_submission_script(script)
        originals = {
            path: path.read_bytes()
            for job in jobs
            for path in (job.paths.submitted_pickle, job.paths.result_pickle)
        }

        mock.sbatch([str(script)])
        for index, original in enumerate(jobs):
            job_id = mock.last_job if count == 1 else f"{mock.last_job}_{index}"
            rerun: slurm.SlurmJob[int] = slurm.SlurmJob(folder=folder, job_id=job_id)
            array_env = (
                {} if count == 1 else {"SLURM_ARRAY_JOB_ID": mock.last_job, "SLURM_ARRAY_TASK_ID": str(index)}
            )
            with mock.job_context(job_id), utils.environment_variables(**array_env):
                _run_submission_script(script)
            assert rerun.result() == original.result() == 3 * (index + 1)
            assert rerun.paths.submitted_pickle.exists()
        assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize("error", [False, True])
def test_slurm_submission_script_reuse_shell(weird_tmp_path: Path, error: bool) -> None:
    if "\n" in str(weird_tmp_path):
        pytest.skip("Slurm directives do not support multiline paths")
    folder = weird_tmp_path / "%j"
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=folder)
        executor.update_parameters(use_srun=False)
        original = executor.submit(test_core.do_nothing, 1, value=2, error=error)
        script = original.paths.submission_file
        original_pickle = original.paths.submitted_pickle.read_bytes()
        mock.sbatch([str(script)])
        rerun: slurm.SlurmJob[int] = slurm.SlurmJob(folder=folder, job_id=mock.last_job)
        rerun.paths.folder.mkdir(parents=True)
        with (
            mock.job_context(rerun.job_id),
            rerun.paths.stdout.open("w") as stdout,
            rerun.paths.stderr.open("w") as stderr,
        ):
            completed = subprocess.run(
                ["bash", str(script)], stdout=stdout, stderr=stderr, timeout=60, check=False
            )
        assert completed.returncode == int(error)
        assert "my args (1,)" in (rerun.stdout() or "")
        assert "'value': 2" in (rerun.stdout() or "")
        if error:
            with pytest.raises(utils.FailedJobError, match="Too bad"):
                rerun.result()
            assert "ValueError: Too bad" in (rerun.stderr() or "")
        else:
            assert rerun.result() == 12
        assert rerun.submission().args == (1,)
        assert original.paths.submitted_pickle.read_bytes() == original_pickle
        assert not original.paths.result_pickle.exists()
        assert not original.paths.stdout.exists()
        assert not original.paths.stderr.exists()


def test_slurm_submission_script_reuse_multitask(tmp_path: Path) -> None:
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path / "%j")
        executor.update_parameters(ntasks_per_node=2)
        original = executor.submit(test_core._three_time, 2)
        script = original.paths.submission_file
        mock.sbatch([str(script)])
        rerun: slurm.SlurmJob[int] = slurm.SlurmJob(executor.folder, mock.last_job, tasks=(0, 1))
        for rank in (0, 1):
            with (
                mock.job_context(rerun.job_id),
                utils.environment_variables(SLURM_PROCID=rank, SLURM_NTASKS=2),
            ):
                _run_submission_script(script)
            assert rerun.paths.submitted_pickle.exists()
        assert rerun.results() == [6, 6]


def test_slurm_submission_script_reuse_shared_snapshot(tmp_path: Path) -> None:
    class Value(helpers.Checkpointable):
        value = 0

        def __call__(self) -> int:
            return self.value

    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path / "%j")
        executor.update_parameters(ntasks_per_node=2)
        original = executor.submit(Value())
        script = original.paths.submission_file
        mock.sbatch([str(script)])
        rerun: slurm.SlurmJob[int] = slurm.SlurmJob(executor.folder, mock.last_job, tasks=(0, 1))

        def start_rank_zero(_: float) -> None:
            delayed = original.submission()
            assert isinstance(delayed.function, Value)
            delayed.function.value = 1
            assert job_environment._checkpoint(delayed, original.paths.submitted_pickle, 3) == ""
            with mock.job_context(rerun.job_id), utils.environment_variables(SLURM_PROCID=0, SLURM_NTASKS=2):
                _run_submission_script(script)

        with (
            mock.job_context(rerun.job_id),
            utils.environment_variables(SLURM_PROCID=1, SLURM_NTASKS=2),
            patch("submitit.core.submission.time.sleep", side_effect=start_rank_zero) as sleep,
        ):
            _run_submission_script(script)
        if not sleep.called:
            start_rank_zero(1)
        assert rerun.results() == [1, 1]
        sleep.assert_called_once_with(1)


@pytest.mark.parametrize("remove_original", [False, True])
@pytest.mark.parametrize("timed_out", [False, True])
def test_slurm_submission_script_reuse_checkpoint(
    tmp_path: Path, remove_original: bool, timed_out: bool
) -> None:
    class Counter(helpers.Checkpointable):
        value = 0

        def __call__(self) -> int:
            self.value += 1
            if self.value == 1:
                signal.raise_signal(submitit.JobEnvironment._usr_sig())
            return self.value

    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path / "%j", max_num_timeout=1)
        executor.update_parameters(time=60)
        original = executor.submit(Counter())
        script = original.paths.submission_file
        original_pickle = original.paths.submitted_pickle.read_bytes()
        mock.sbatch([str(script)])
        rerun: slurm.SlurmJob[int] = slurm.SlurmJob(executor.folder, mock.last_job)
        with mock.job_context(rerun.job_id):
            with (
                pytest.raises(SystemExit),
                mock_requeue(called_with=1 - timed_out),
                patch.object(job_environment.SignalHandler, "has_timed_out", return_value=timed_out),
            ):
                _run_submission_script(script)
            checkpoint = rerun.submission()
            assert isinstance(checkpoint.function, Counter)
            assert checkpoint.function.value == 1
            assert checkpoint._timeout_countdown == 1 - timed_out
            assert original.paths.submitted_pickle.read_bytes() == original_pickle
            if remove_original:
                original.paths.submitted_pickle.unlink()
            _run_submission_script(script)
        assert rerun.result() == 2


@pytest.mark.parametrize("publish_current", [False, True])
def test_slurm_submission_script_reuse_wait(tmp_path: Path, publish_current: bool) -> None:
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path / "%j")
        original = executor.submit(test_core._three_time, 3)
        script = original.paths.submission_file
        source = next(tmp_path.glob("*.submitted.pkl"))
        source.unlink()
        mock.sbatch([str(script)])
        rerun: slurm.SlurmJob[int] = slurm.SlurmJob(executor.folder, mock.last_job)

        def publish(_: float) -> None:
            if publish_current:
                rerun.paths.folder.mkdir(parents=True)
                utils.DelayedSubmission(test_core._three_time, 4).dump(rerun.paths.submitted_pickle)
            else:
                source.symlink_to(original.paths.submitted_pickle)

        with (
            mock.job_context(rerun.job_id),
            patch("submitit.core.submission.time.sleep", side_effect=publish) as sleep,
        ):
            _run_submission_script(script)
        sleep.assert_called_once_with(1)
        assert rerun.result() == (12 if publish_current else 9)


def test_slurm_submission_script_reuse_copy_error(tmp_path: Path) -> None:
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path / "%j")
        original = executor.submit(test_core._three_time, 3)
        script = original.paths.submission_file
        original_pickle = original.paths.submitted_pickle.read_bytes()
        mock.sbatch([str(script)])
        rerun: slurm.SlurmJob[int] = slurm.SlurmJob(executor.folder, mock.last_job)

        def fail_copy(_: Path, destination: Path) -> None:
            destination.write_bytes(b"partial")
            raise OSError("copy failed")

        with (
            mock.job_context(rerun.job_id),
            patch("submitit.core.submission.shutil.copyfile", side_effect=fail_copy),
        ):
            with pytest.raises(OSError, match="copy failed"):
                _run_submission_script(script)
        with pytest.raises(utils.FailedJobError, match="copy failed"):
            rerun.result()
        assert not rerun.paths.submitted_pickle.exists()
        assert not list(rerun.paths.folder.glob("*.save_tmp"))
        assert original.paths.submitted_pickle.read_bytes() == original_pickle


@pytest.mark.parametrize("use_batch_api", (False, True))  # type: ignore
def test_slurm_job_array_mocked(use_batch_api: bool, tmp_path: Path) -> None:
    n = 5
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path / "%j")
        executor.update_parameters(array_parallelism=3)
        data1, data2 = range(n), range(10, 10 + n)

        def add(x: int, y: int) -> int:
            assert x in data1
            assert y in data2
            return x + y

        jobs: list[Job[int]] = []
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
        # trying a slurm specific method
        jobs[0]._interrupt(timeout=True)  # type: ignore
        assert list(map(add, data1, data2)) == [j.result() for j in jobs]
        # check submission file
        sbatch = jobs[0].paths.submission_file.read_text()
        array_line = [l.strip() for l in sbatch.splitlines() if "--array" in l]
        assert array_line == ["#SBATCH --array=0-4%3"]


def test_slurm_error_mocked(tmp_path: Path) -> None:
    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path)
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
def mock_requeue(called_with: int | None = None, not_called: bool = False):
    assert not_called or called_with is not None
    requeue = patch("submitit.slurm.slurm.SlurmJobEnvironment._requeue", return_value=None)
    with requeue as _patch:
        try:
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
    usr_sig = submitit.JobEnvironment._usr_sig()
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
    usr_sig = slurm.SlurmJobEnvironment._usr_sig()
    with (
        mock_requeue(not_called=True),
        pytest.raises(utils.UncompletedJobError, match="timed-out too many times."),
    ):
        sig.checkpoint_and_try_requeue(usr_sig)


@pytest.mark.parametrize("time", [60, "1:00:00"])
def test_requeuing_not_checkpointable(
    tmp_path: Path, fast_forward_clock, monkeypatch, time: int | str
) -> None:
    monkeypatch.setenv("SLURM_JOB_START_TIME", "0")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "3600" if isinstance(time, str) else "60")  # numeric time wins
    usr_sig = submitit.JobEnvironment._usr_sig()
    # Start job with a 60 minutes timeout
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=time)
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
    with (
        mock_requeue(not_called=True),
        pytest.raises(utils.UncompletedJobError, match="timed-out and not checkpointable"),
    ):
        sig.checkpoint_and_try_requeue(usr_sig)


def test_checkpoint_and_exit(tmp_path: Path) -> None:
    usr_sig = submitit.JobEnvironment._usr_sig()
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=tmp_path, max_num_timeout=1)
        executor.update_parameters(time=60)
        job = executor.submit(test_core._three_time, 10)

    sig = get_signal_handler(job)
    with pytest.raises(SystemExit), mock_requeue(not_called=True):
        sig.checkpoint_and_exit(usr_sig)

    # checkpoint_and_exit doesn't modify timeout counters.
    delayed = utils.DelayedSubmission.load(job.paths.submitted_pickle)
    assert delayed._timeout_countdown == 1


def test_make_sbatch_string() -> None:
    string = slurm._make_sbatch_string(
        command="blublu bar",
        folder="/tmp",
        partition="learnfair",
        exclusive=True,
        additional_parameters={"blublu": 12},
        srun_args=["-vv", "--cpu-bind", "none"],
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


def test_make_sbatch_string_gpu() -> None:
    string = slurm._make_sbatch_string(command="blublu", folder="/tmp", gpus_per_node=2, wckey="")
    assert "--gpus-per-node=2" in string
    assert "--wckey" not in string, "empty options must be removed"


def test_make_sbatch_stderr() -> None:
    string = slurm._make_sbatch_string(command="blublu", folder="/tmp", stderr_to_stdout=True)
    assert "--error" not in string


def test_update_parameters(tmp_path: Path) -> None:
    with mocked_slurm():
        executor = submitit.AutoExecutor(folder=tmp_path)
    executor.update_parameters(mem_gb=3.5)
    assert executor._executor.parameters["mem"] == "3584MB"


def test_update_parameters_error(tmp_path: Path) -> None:
    with mocked_slurm():
        executor = slurm.SlurmExecutor(folder=tmp_path)
    with pytest.raises(ValueError):
        executor.update_parameters(blublu=12)


def test_read_info() -> None:
    example = """JobID|State
5610980|RUNNING
5610980.ext+|RUNNING
5610980.0|RUNNING
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
def test_read_job_id(job_id: str, expected: list[tuple[int | str, ...]]) -> None:
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
    with mocked_slurm() as mock:
        watcher = slurm.SlurmInfoWatcher()
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
def test_slurm_through_auto(params: dict[str, int], tmp_path: Path) -> None:
    with mocked_slurm():
        executor = submitit.AutoExecutor(folder=tmp_path)
        executor.update_parameters(**params, slurm_additional_parameters={"mem_per_gpu": 12})
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    mem_lines = [x for x in text.splitlines() if "#SBATCH --mem" in x]
    assert len(mem_lines) == 1, f"Unexpected lines: {mem_lines}"


def test_slurm_job_no_stderr(tmp_path: Path) -> None:
    def fail_silently():
        raise ValueError("Too bad")

    with mocked_slurm() as mock:
        executor = slurm.SlurmExecutor(folder=tmp_path)
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
        # Explicitly unlink stdout because submitit is writing there on startup
        # job.paths.stdout.unlink()
        with pytest.raises(utils.UncompletedJobError, match="No output/error stream produced !"):
            job._get_outcome_and_result()


def test_slurm_through_auto_with_setup_and_teardown(tmp_path: Path) -> None:
    with mocked_slurm():
        executor = submitit.AutoExecutor(folder=tmp_path)
        executor.update_parameters(
            slurm_additional_parameters={"mem_per_gpu": 12},
            slurm_setup=["This is a setup command"],
            slurm_teardown=["This is a teardown command"],
        )
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    assert "This is a setup command" in text
    assert "This is a teardown command" in text
