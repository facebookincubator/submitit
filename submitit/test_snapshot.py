import os
from pathlib import Path

from .local import local
from .snapshot import SnapshotManager


def test_snapshot(tmp_path):
    cwd = Path.cwd()
    with SnapshotManager(tmp_path):
        assert Path.cwd() == tmp_path
        assert (tmp_path / "submitit/test_snapshot.py").exists()
    assert Path.cwd() == cwd


def test_exclude(tmp_path):
    exclude = ["submitit/test_*"]
    with SnapshotManager(snapshot_dir=tmp_path, exclude=exclude):
        assert (tmp_path / "submitit/snapshot.py").exists()
        assert not (tmp_path / "submitit/test_snapshot.py").exists()


def test_submitted_job(tmp_path):
    executor = local.LocalExecutor(tmp_path)
    with SnapshotManager(snapshot_dir=tmp_path):
        job = executor.submit(os.getcwd)
        assert Path(job.result()) == tmp_path
