# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path

import pytest

from . import submission, utils
from .test_core import FakeExecutor, FakeJob


class TaggedError(Exception):
    code = 42


class UnpicklableError(Exception):
    def __reduce__(self):
        raise TypeError("cannot pickle this exception")


class BadInitError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(kind: str) -> None:
    if kind == "custom":
        error = TaggedError("remote failure")
        error.code = 73
        raise error
    if kind == "dump":
        raise UnpicklableError("remote failure")
    if kind == "load":
        raise BadInitError(42, "remote failure")
    raise ValueError("remote failure")


@pytest.mark.parametrize("kind, error_type", [("builtin", ValueError), ("custom", TaggedError)])
def test_original_exception_cause(tmp_path: Path, kind: str, error_type: type[Exception]) -> None:
    job = FakeExecutor(folder=tmp_path).submit(_fail, kind)
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        with pytest.raises(error_type, match="remote failure"):
            submission.process_job(job.paths.folder)
    error = job.exception()
    assert isinstance(error, utils.FailedJobError)
    assert isinstance(error.__cause__, error_type)
    assert str(error.__cause__) == "remote failure"
    if isinstance(error.__cause__, TaggedError):
        assert error.__cause__.code == 73
    with pytest.raises(utils.FailedJobError) as caught:
        job.result()
    assert isinstance(caught.value.__cause__, error_type)


@pytest.mark.parametrize("kind", ["dump", "load"])
def test_original_exception_fallback(tmp_path: Path, kind: str) -> None:
    job = FakeExecutor(folder=tmp_path).submit(_fail, kind)
    with utils.environment_variables(_TEST_CLUSTER_="slurm", SLURM_JOB_ID=str(job.job_id)):
        with pytest.raises(Exception, match="remote failure"):
            submission.process_job(job.paths.folder)
    error = job.exception()
    assert isinstance(error, utils.FailedJobError)
    assert error.__cause__ is None
    assert "remote failure" in str(error)


def test_legacy_error_result(tmp_path: Path) -> None:
    job = FakeJob[None](job_id="12", folder=tmp_path)
    utils.cloudpickle_dump(("error", "legacy traceback"), job.paths.result_pickle)
    error = job.exception()
    assert isinstance(error, utils.FailedJobError)
    assert "legacy traceback" in str(error)
    assert error.__cause__ is None


@pytest.mark.parametrize(
    "error_pickle", [b"invalid pickle", pickle.dumps(BadInitError(42, "failure")), pickle.dumps("text")]
)
def test_invalid_exception_pickle_fallback(tmp_path: Path, error_pickle: bytes) -> None:
    job = FakeJob[None](job_id="12", folder=tmp_path)
    utils.cloudpickle_dump(("error", ("remote traceback", error_pickle)), job.paths.result_pickle)
    error = job.exception()
    assert isinstance(error, utils.FailedJobError)
    assert "remote traceback" in str(error)
    assert error.__cause__ is None
