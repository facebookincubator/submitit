# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from . import utils


def _read(filepath: Path) -> Optional[str]:
    """Returns content if file exists
    """
    if not filepath.exists():
        return None
    with filepath.open("r") as f:
        text: str = f.read()
    return text.strip()


@pytest.mark.parametrize("existing_content", [None, "blublu"])  # type: ignore
def test_temporary_save_path(existing_content: Optional[str]) -> None:
    with tempfile.TemporaryDirectory() as folder:
        filepath = Path(folder) / "save_and_move_test.txt"
        if existing_content is not None:
            with filepath.open("w") as f:
                f.write(existing_content)
        with utils.temporary_save_path(filepath) as tmp:
            assert str(tmp).endswith(".txt.save_tmp"), f"Unexpected path {tmp}"
            with tmp.open("w") as f:
                f.write("12")
            assert _read(filepath) == existing_content
        assert _read(filepath) == "12"


def test_temporary_save_path_error() -> None:
    with pytest.raises(FileNotFoundError):
        with utils.temporary_save_path("save_and_move_test"):
            pass


def _three_time(x: int) -> int:
    return 3 * x


def test_delayed() -> None:
    delayed = utils.DelayedSubmission(_three_time, 4)
    assert not delayed.done()
    assert delayed.result() == 12
    assert delayed.done()
    with tempfile.TemporaryDirectory() as folder:
        filepath = Path(folder) / "test_delayed.pkl"
        delayed.dump(filepath)
        delayed2 = utils.DelayedSubmission.load(filepath)
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


def test_sanitize() -> None:
    assert utils.sanitize("AlreadySanitized") == "AlreadySanitized"
    assert utils.sanitize("Name with space") == "Name_with_space"
    assert utils.sanitize("Name with space", only_alphanum=False) == '"Name with space"'
    assert utils.sanitize("Name with    many    spaces") == "Name_with_many_spaces"
    assert utils.sanitize(" Non alph@^ Nüm%") == "_Non_alph_Nüm_"


def test_archive_dev_folders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_dev_folders_"
        utils.archive_dev_folders([Path(__file__).parent], outfile=path.with_suffix(".tar.gz"))
        shutil.unpack_archive(str(path.with_suffix(".tar.gz")), extract_dir=path)
        expected = path / Path(__file__).parent.name
        assert (
            expected.exists()
        ), f"Missing submitit folder (expected {expected} but found: {list(path.iterdir())})"


def test_command_function() -> None:
    command = f"{sys.executable} -m submitit.core.test_core".split()
    word = "testblublu12"
    output = utils.CommandFunction(command)(word)
    assert output is not None
    assert word in output, f'Missing word "{word}" in output:\n{output}'
    try:
        with contextlib.redirect_stderr(sys.stdout):
            output = utils.CommandFunction(command, verbose=True)(error=True)
    except utils.FailedJobError as e:
        words = "Too bad"
        assert words in str(e), f'Missing word "{words}" in output:\n\n{e}'
    else:
        raise AssertionError("An error should have been raised")
