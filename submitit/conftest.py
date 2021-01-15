from pathlib import Path

import pytest

from .local.local import LocalExecutor


@pytest.fixture()
def executor(tmp_path: Path) -> LocalExecutor:
    return LocalExecutor(tmp_path)
