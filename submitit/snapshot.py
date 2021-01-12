import fnmatch
import os
import shutil
from subprocess import check_output


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
        # https://stackoverflow.com/a/51689219/4876946
        files = run_cmd(f"git ls-files {sub} | grep -v ^16 | cut -f2-", cwd=root_dir).split("\n")
        # Copy all the files into the snapshot directory
        for file in files:
            if any([fnmatch.fnmatch(file, ex) for ex in self.exclude]):
                continue
            dest_file = os.path.join(self.snapshot_dir, file)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copy(os.path.join(root_dir, file), dest_file)
        os.chdir(self.snapshot_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)
