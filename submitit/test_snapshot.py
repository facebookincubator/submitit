import os
from tempfile import TemporaryDirectory

from .snapshot import SnapshotManager


def test_snapshot():
    cwd = os.getcwd()
    with TemporaryDirectory() as tdir:
        with SnapshotManager(tdir):
            assert os.getcwd() == tdir
            assert os.path.exists(os.path.join(tdir, "submitit/test_snapshot.py"))
    assert os.getcwd() == cwd
