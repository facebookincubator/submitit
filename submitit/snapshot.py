import fnmatch
import os
import shutil
from subprocess import check_output
from tempfile import NamedTemporaryFile

def run_cmd(str_args, **kwargs):
    return check_output(str_args, shell=True, **kwargs).decode("utf-8").strip()


class SnapshotManager:
    def __init__(
        self, snapshot_dir: str, with_submodules: bool = False, exclude=None,
    ):
        self.snapshot_dir = snapshot_dir
        self.original_dir = os.getcwd()
        self.with_submodules = with_submodules
        self.exclude = exclude or []

    def __enter__(self):
        self.original_dir = os.getcwd()
        os.makedirs(self.snapshot_dir, exist_ok=True)
        # Get the repository root
        root_dir = run_cmd("git rev-parse --show-toplevel")
        sub = "--recurse-submodules" if self.with_submodules else "-s"
        # Get a list of all the checked in files that we can pass to rsync
        with NamedTemporaryFile() as tfile:
            # https://stackoverflow.com/a/51689219/4876946
            files = run_cmd(f"git ls-files {sub} | grep -v ^16 | cut -f2- > {tfile.name}", cwd=root_dir).split("\n")
            run_cmd(f"rsync -a --files-from={tfile.name} {root_dir} {self.snapshot_dir}")
        os.chdir(self.snapshot_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)
