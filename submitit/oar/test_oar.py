# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import os
import subprocess
import typing as tp
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence
from unittest.mock import patch

import pytest

import submitit

from ..core import job_environment, submission, test_core, utils
from ..core.core import Job
from . import oar


# pylint: disable=no-self-use
class MockedSubprocess:
    """Helper for mocking subprocess calls"""

    OARSTAT_JOB = '{{"{j}" : {{"state" : "{state}"}}}}'

    def __init__(self, known_cmds: Sequence[str] = None) -> None:
        self.job_oarstat: Dict[str, str] = {}
        self.last_job: str = ""
        self._subprocess_check_output = subprocess.check_output
        self.known_cmds = known_cmds or []
        self.job_count = 12

    def __call__(self, command: Sequence[str], **kwargs: Any) -> bytes:
        program = command[0]
        if program in ["oarstat", "oarsub", "oardel"]:
            return getattr(self, program)(command[1:]).encode()
        elif program == "tail":
            return self._subprocess_check_output(command, **kwargs)
        else:
            raise ValueError(f'Unknown command to mock "{command}".')

    def oarstat(self, _: Sequence[str]) -> str:
        return "\n".join(self.job_oarstat.values())

    def oarsub(self, args: Sequence[str]) -> str:
        """Create a "Running" job."""
        job_id = str(self.job_count)
        self.job_count += 1
        self.set_job_state(job_id, "Running", 0)
        return f"OAR_JOB_ID={job_id}\n"

    def oardel(self, _: Sequence[str]) -> str:
        # TODO:should we call set_job_state ?
        return ""

    def set_job_state(self, job_id: str, state: str, array: int = 0) -> None:
        self.job_oarstat[job_id] = self._oarstat(state, job_id, array)
        self.last_job = job_id

    def _oarstat(self, state: str, job_id: str, array: int) -> str:
        if array == 0:
            lines = self.OARSTAT_JOB.format(j=job_id, state=state)
        else:
            lines = "\n".join(self.OARSTAT_JOB.format(j=f"{job_id}_{i}", state=state) for i in range(array))
        return lines

    def which(self, name: str) -> Optional[str]:
        return "here" if name in self.known_cmds else None

    def mock_cmd_fn(self, *args, **_):
        # CommandFunction(cmd)() ~= subprocess.check_output(cmd)
        return lambda: self(*args)

    @contextlib.contextmanager
    def context(self) -> Iterator[None]:
        with patch("submitit.core.utils.CommandFunction", new=self.mock_cmd_fn):
            with patch("subprocess.check_output", new=self):
                with patch("shutil.which", new=self.which):
                    with patch("subprocess.check_call", new=self):
                        yield None

    @contextlib.contextmanager
    def job_context(self, job_id: str) -> Iterator[None]:
        with utils.environment_variables(
            _USELESS_TEST_ENV_VAR_="1", SUBMITIT_EXECUTOR="oar", OAR_JOB_ID=str(job_id)
        ):
            yield None


def _mock_log_files(job: Job[tp.Any], prints: str = "", errors: str = "") -> None:
    """Write fake log files"""
    filepaths = [str(x).replace("%j", str(job.job_id)) for x in [job.paths.stdout, job.paths.stderr]]
    for filepath, msg in zip(filepaths, (prints, errors)):
        with Path(filepath).open("w") as f:
            f.write(msg)


@contextlib.contextmanager
def mocked_oar() -> tp.Iterator[MockedSubprocess]:
    mock = MockedSubprocess(known_cmds=["oarsub"])
    try:
        with mock.context():
            yield mock
    finally:
        # Clear the state of the shared watcher
        oar.OarJob.watcher.clear()


def test_mocked_missing_state(tmp_path: Path) -> None:
    with mocked_oar() as mock:
        mock.set_job_state("12", "")
        job: oar.OarJob[None] = oar.OarJob(tmp_path, "12")
        assert job.state == "UNKNOWN"
        job._interrupt(timeout=False)  # check_call is bypassed by MockedSubprocess


def test_job_environment() -> None:
    with mocked_oar() as mock:
        mock.set_job_state("12", "Running")
        with mock.job_context("12"):
            assert job_environment.JobEnvironment().cluster == "oar"


def test_oar_job_mocked(tmp_path: Path) -> None:
    with mocked_oar() as mock:
        executor = oar.OarExecutor(folder=tmp_path)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
        # First mock job always have id 12
        assert job.job_id == "12"
        assert job.state == "Running"
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


def test_oar_error_mocked(tmp_path: Path) -> None:
    with mocked_oar() as mock:
        executor = oar.OarExecutor(folder=tmp_path)
        executor.update_parameters(walltime="0:0:5", queue="default")  # just to cover the function
        job = executor.submit(test_core.do_nothing, 1, 2, error=12)
        with mock.job_context(job.job_id):
            with pytest.raises(ValueError):
                submission.process_job(job.paths.folder)
        _mock_log_files(job, errors="This is the error log\n")
        with pytest.raises(utils.FailedJobError):
            job.result()
        exception = job.exception()
        assert isinstance(exception, utils.FailedJobError)


def test_make_oarsub_string() -> None:
    string = oar._make_oarsub_string(
        command="blublu bar",
        folder="/tmp",
        queue="default",
        additional_parameters=dict({"t": "besteffort"}),
    )
    assert "q" in string
    assert "-t besteffort" in string
    assert "nodes" not in string
    assert "core" not in string
    assert "gpu" not in string
    assert "--command" not in string
    record_file = Path(__file__).parent / "_oarsub_test_record.txt"
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


def test_make_oarsub_string_gpu() -> None:
    string = oar._make_oarsub_string(command="blublu", folder="/tmp", gpu=2)
    assert "-l /gpu=2" in string


def test_make_oarsub_string_core() -> None:
    string = oar._make_oarsub_string(command="blublu", folder="/tmp", core=2)
    assert "-l /core=2" in string


def test_make_oarsub_string_gpu_and_nodes() -> None:
    string = oar._make_oarsub_string(command="blublu", folder="/tmp", gpu=2, nodes=1)
    assert "-l /nodes=1/gpu=2" in string


def test_make_oarsub_string_core_and_nodes() -> None:
    string = oar._make_oarsub_string(command="blublu", folder="/tmp", core=2, nodes=1)
    assert "-l /nodes=1/core=2" in string


def test_make_oarsub_string_core_gpu_and_nodes() -> None:
    string = oar._make_oarsub_string(command="blublu", folder="/tmp", gpu=2, nodes=1, core=4)
    assert "-l /nodes=1/gpu=2/core=4" in string


def test_update_parameters(tmp_path: Path) -> None:
    with mocked_oar():
        executor = submitit.AutoExecutor(folder=tmp_path)
    executor.update_parameters(oar_walltime="2:0:0")
    assert executor._executor.parameters["walltime"] == "2:0:0"


def test_update_parameters_error(tmp_path: Path) -> None:
    with mocked_oar():
        executor = oar.OarExecutor(folder=tmp_path)
    with pytest.raises(ValueError):
        executor.update_parameters(blublu=12)


def test_read_info() -> None:
    example = """{
        "1924697" : {
            "state" : "Running"
        }
    }"""
    output = oar.OarInfoWatcher().read_info(example)
    assert output["1924697"] == {"JobID": '1924697', "NodeList" : None, "State" : 'Running'}


def test_watcher() -> None:
    with mocked_oar() as mock:
        watcher = oar.OarInfoWatcher()
        mock.set_job_state("12", "Running")
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
    defaults = oar._get_default_parameters()
    assert defaults["n"] == "submitit"


def test_name() -> None:
    assert oar.OarExecutor.name() == "oar"


@contextlib.contextmanager
def with_oar_job_nodefile(node_list: str) -> tp.Iterator[oar.OarJobEnvironment]:
    node_file_path = Path(__file__).parent / "_oar_node_file.txt"
    _mock_oar_node_file(node_file_path, node_list)
    os.environ["OAR_JOB_ID"] = "1"
    os.environ["OAR_NODEFILE"] = str(Path.joinpath(node_file_path))
    yield oar.OarJobEnvironment()
    del os.environ["OAR_NODEFILE"]
    del os.environ["OAR_JOB_ID"]


def _mock_oar_node_file(node_file_path, node_list: str) -> None:
    """Write fake oar node file"""
    with open(node_file_path, "w+") as file:
        file.write(node_list)


def test_oar_node_file() -> None:
    with with_oar_job_nodefile('chetemi-7.lille.grid5000.fr\n') as env:
        assert env.hostnames == ['chetemi-7.lille.grid5000.fr']
        assert env.num_nodes == 1
        assert env.node == 0
    with with_oar_job_nodefile('chetemi-8.lille.grid5000.fr\nchetemi-8.lille.grid5000.fr\nchetemi-7.lille.grid5000.fr\nchetemi-7.lille.grid5000.fr\n') as env:
        assert ["chetemi-7.lille.grid5000.fr", "chetemi-8.lille.grid5000.fr"] == env.hostnames
        assert env.num_nodes == 2
        assert env.node == 0


@pytest.mark.parametrize("params", [{}, {"timeout_min": None}])  # type: ignore
def test_oar_through_auto(params: tp.Dict[str, int], tmp_path: Path) -> None:
    with mocked_oar():
        executor = submitit.AutoExecutor(folder=tmp_path)
        executor.update_parameters(**params, oar_additional_parameters={"t": 'besteffort'})
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    best_effort_lines = [x for x in text.splitlines() if "#OAR -t besteffort" in x]
    assert len(best_effort_lines) == 1, f"Unexpected lines: {best_effort_lines}"


def test_timeout_min_to_oar_walltime(tmp_path: Path) -> None:
    with mocked_oar():
        executor = submitit.AutoExecutor(folder=tmp_path)
        executor.update_parameters(timeout_min=90)
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    walltime_lines = [x for x in text.splitlines() if "#OAR -l walltime=01:30" in x]
    assert len(walltime_lines) == 1, f"Unexpected lines: {walltime_lines}"


def test_oar_walltime_wins_over_timeout_min(tmp_path: Path) -> None:
    with mocked_oar():
        executor = submitit.AutoExecutor(folder=tmp_path)
        executor.update_parameters(timeout_min=90, oar_walltime="2:0:0")
        job = executor.submit(test_core.do_nothing, 1, 2, blublu=3)
    text = job.paths.submission_file.read_text()
    walltime_lines = [x for x in text.splitlines() if "#OAR -l walltime=2:0:0" in x]
    assert len(walltime_lines) == 1, f"Unexpected lines: {walltime_lines}"
