# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest

from . import utils


@pytest.mark.parametrize("existing_content", [None, "blublu"])  # type: ignore
def test_temporary_save_path(tmp_path: Path, existing_content: Optional[str]) -> None:
    filepath = tmp_path / "save_and_move_test.txt"
    if existing_content:
        filepath.write_text(existing_content)
    with utils.temporary_save_path(filepath) as tmp:
        assert str(tmp).endswith(".txt.save_tmp")
        tmp.write_text("12")
        if existing_content:
            assert filepath.read_text() == existing_content
    assert filepath.read_text() == "12"


def test_temporary_save_path_preserves_destination_until_replace(tmp_path: Path, monkeypatch) -> None:
    filepath = tmp_path / "checkpoint.pkl"
    filepath.write_text("previous checkpoint")
    replace = os.replace
    observed = []

    def observe_move(source, destination):
        # Observe the actual destination immediately before the native move.
        observed.append(Path(destination).read_text())
        return replace(source, destination)

    monkeypatch.setattr(os, "rename", observe_move)
    monkeypatch.setattr(os, "replace", observe_move)
    with utils.temporary_save_path(filepath) as tmp:
        tmp.write_text("next checkpoint")
    assert observed == ["previous checkpoint"]
    assert filepath.read_text() == "next checkpoint"
    assert not tmp.exists()


@pytest.mark.parametrize("existing_content", [None, "previous checkpoint"])
def test_temporary_save_path_move_failure(
    tmp_path: Path, monkeypatch, existing_content: Optional[str]
) -> None:
    filepath = tmp_path / "checkpoint.pkl"
    if existing_content is not None:
        filepath.write_text(existing_content)

    def fail_move(_source, _destination):
        raise OSError("injected move failure")

    monkeypatch.setattr(os, "rename", fail_move)
    monkeypatch.setattr(os, "replace", fail_move)
    with pytest.raises(OSError, match="injected move failure"):
        with utils.temporary_save_path(filepath) as tmp:
            tmp.write_text("next checkpoint")
    if existing_content is None:
        assert not filepath.exists()
    else:
        assert filepath.read_text() == existing_content


def test_temporary_save_path_body_failure(tmp_path: Path) -> None:
    filepath = tmp_path / "checkpoint.pkl"
    filepath.write_text("previous checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        with utils.temporary_save_path(filepath) as tmp:
            tmp.write_text("partial checkpoint")
            raise RuntimeError("checkpoint failed")
    assert filepath.read_text() == "previous checkpoint"


def test_temporary_save_path_error() -> None:
    with pytest.raises(FileNotFoundError):
        with utils.temporary_save_path("save_and_move_test"):
            pass


def _three_time(x: int) -> int:
    return 3 * x


def test_delayed(tmp_path: Path) -> None:
    delayed = utils.DelayedSubmission(_three_time, 4)
    assert not delayed.done()
    assert delayed.result() == 12
    assert delayed.done()
    delayed_pkl = tmp_path / "test_delayed.pkl"
    delayed.dump(delayed_pkl)
    delayed2 = utils.DelayedSubmission.load(delayed_pkl)
    assert delayed2.done()


def test_environment_variable_context() -> None:
    name = "ENV_VAR_TEST"
    assert name not in os.environ
    with utils.environment_variables(ENV_VAR_TEST="blublu"):
        assert os.environ[name] == "blublu"
        with utils.environment_variables(ENV_VAR_TEST="blublu2"):
            assert os.environ[name] == "blublu2"
        assert os.environ[name] == "blublu"
    assert name not in os.environ


def test_slurmpaths_id_independent() -> None:
    path = "test/truc/machin_%j/name"
    output = utils.JobPaths.get_first_id_independent_folder(path)
    assert output.name == "truc"


def test_archive_dev_folders(tmp_path: Path) -> None:
    utils.archive_dev_folders([Path(__file__).parent], outfile=tmp_path.with_suffix(".tar.gz"))
    shutil.unpack_archive(str(tmp_path.with_suffix(".tar.gz")), extract_dir=tmp_path)
    assert (tmp_path / "core").exists()


def test_command_function() -> None:
    # This will call `submitit.core.test_core.do_nothing`
    command = [sys.executable, "-m", "submitit.core.test_core"]
    word = "testblublu12"
    output = utils.CommandFunction(command)(word)
    assert output is not None
    assert word in output
    with pytest.raises(utils.FailedJobError, match="Too bad"):
        # error=True will make `do_nothing` fail
        utils.CommandFunction(command, verbose=True)(error=True)


def test_command_function_deadlock(executor) -> None:
    code = """
import sys;
print(sys.__stderr__)
# The goal here is to fill up the stderr pipe buffer.
for i in range({n}):
    print("-" * 1024, file=sys.stdout)
print("printed {n} lines to stderr")
"""
    fn1 = utils.CommandFunction([sys.executable, "-c", code.format(n=10)])
    executor.update_parameters(timeout_min=2 / 60)
    j1 = executor.submit(fn1)
    assert "10 lines" in j1.result()

    fn2 = utils.CommandFunction(["python", "-c", code.format(n=1000)])
    j2 = executor.submit(fn2)
    assert "1000 lines" in j2.result()


def test_jobpaths(tmp_path: Path) -> None:
    assert utils.JobPaths(tmp_path, "123").stdout == tmp_path / "123_0_log.out"
    assert utils.JobPaths(tmp_path, "123", 1).stdout == tmp_path / "123_1_log.out"
    assert (
        utils.JobPaths(tmp_path / "array-%A-index-%a", "456_3").stdout
        == tmp_path / "array-456-index-3" / "456_3_0_log.out"
    )
