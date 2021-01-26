from pathlib import Path

import pytest

from .local.local import LocalExecutor


@pytest.fixture()
def executor(tmp_path: Path) -> LocalExecutor:
    return LocalExecutor(tmp_path)


@pytest.fixture(params=["a_0", "a 0", 'a"=0"', "a'; echo foo", r"a\=0", r"a\=", "a\n0"])
def weird_dir(request) -> str:
    return request.param
