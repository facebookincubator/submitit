import itertools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import shutil


def run_cmd(str_args, **kwargs):
    return subprocess.check_output(str_args, **kwargs).decode("utf-8").strip()


class SnapshotManager:
    """Snapshot Manager
    This class creates a snapshot of the git repository that the script lives in
    when creating the snapshot.  This is useful for ensuring that remote jobs that
    get launched don't accidentally pick up unintended local changes.

    Parameters
    ----------
    snapshot_dir: Path
        A path to where the snapshot should be created
    with_submodules: bool
        Whether or not submodules should be included in the snapshot
    exclude: Optional[List[str]]
        An optional list of patterns to exclude from the snapshot

    Note
    ----
    - Only files that are checked in to the repository are included in the snapshot.
        If you have experimental code that you would like to include in the snapshot,
        you'll need to `git add` the file first for it to be included
    """

    def __init__(
        self, snapshot_dir: Path, with_submodules: bool = False, exclude: Optional[List[str]] = None,
    ):
        if shutil.which('rsync') is None:
            raise RuntimeError("SnapshotManager requires rsync to be installed.")
        self.snapshot_dir = snapshot_dir
        self.original_dir = Path.cwd()
        self.with_submodules = with_submodules
        self.exclude = exclude or []

    def __enter__(self):
        self.original_dir = Path.cwd()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Get the repository root
        root_dir = run_cmd(["git", "rev-parse", "--show-toplevel"])
        sub = "--recurse-submodules" if self.with_submodules else "-s"
        # Get a list of all the checked in files that we can pass to rsync
        with tempfile.NamedTemporaryFile() as tfile:
            # https://stackoverflow.com/a/51689219/4876946
            run_cmd(f"git ls-files {sub} | grep -v ^16 | cut -f2- > {tfile.name}", cwd=root_dir, shell=True)
            exclude = list(itertools.chain(*[["--exclude", pat] for pat in self.exclude]))
            run_cmd(["rsync", "-a", f"--files-from={tfile.name}", root_dir, self.snapshot_dir] + exclude)
        os.chdir(self.snapshot_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)
