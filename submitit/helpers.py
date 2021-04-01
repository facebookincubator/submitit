# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import os
import shutil
import subprocess
import tempfile
import time
import typing as tp
from pathlib import Path

# pylint: disable=unused-import
# import DelayedSubmission and CommandFunction to populate helpers namespace
from .core import core
from .core.utils import CommandFunction as CommandFunction  # noqa
from .core.utils import DelayedSubmission as DelayedSubmission  # noqa


class Checkpointable:
    """Derived callable classes are requeued after timeout with their current
    state dumped at checkpoint.

    __call__ method must be implemented to make your class a callable.

    Note
    ----
    The following implementation of the checkpoint method resubmits the full current
    state of the callable (self) with the initial argument. You may want to replace the method to
    curate the state (dump a neural network to a standard format and remove it from
    the state so that not to pickle it) and change/remove the initial parameters.
    """

    # pylint: disable=unused-argument
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        assert callable(
            instance
        ), f"Class {cls.__name__} is marked as Checkpointable but doesn't have a __call__ method. Please add a __call__ method."
        return instance

    def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> DelayedSubmission:
        """Resubmits the same callable with the same arguments"""
        # The DelayedSubmission class goal is only to register and format
        # the arguments of the call "self(*args, **kwargs)" for submission to slurm
        return DelayedSubmission(self, *args, **kwargs)  # type: ignore


class FunctionSequence(Checkpointable):
    """This is for gathering several estimations into one function, which
    will return the sequence of outputs.
    Also this "function" is stateful, hence it can be stopped, and recovered,
    which is useful when job can be preempted.

    Usage
    -----
    func = FunctionSequence()
    func.add(my_function1, arg1, kwarg1=value_kwarg1)
    func.add(my_function2, arg1, arg2)
    result1, result2 = func()

    Note
    ----
    This function is checkpointable because:
    - it derives from Checkpointable
    - it keeps DelayedSubmission objects as attribute, which in turn store the
      results of the computation in memory once they are computed. So at checkpoint
      time, those results will be saved, and only the non-computed results
      will be computed once the job restarts.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.delayed_functions: tp.List[DelayedSubmission] = []

    def add(self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        self.delayed_functions.append(DelayedSubmission(func, *args, **kwargs))

    def __len__(self) -> int:
        return len(self.delayed_functions)

    def __iter__(self) -> tp.Iterator[DelayedSubmission]:
        return iter(self.delayed_functions)

    def __call__(self) -> tp.List[tp.Any]:  # pylint: disable=arguments-differ
        if self.verbose:
            done = sum(f.done() for f in self)  # those were computed before checkpoint
            print(f"Starting from {done}/{len(self.delayed_functions)}", flush=True)
        return [
            f.result() for f in self.delayed_functions
        ]  # results all results one by one (by running the functions if not already done)


def as_completed(
    jobs: tp.Sequence[core.Job[core.R]],
    timeout: tp.Optional[tp.Union[int, float]] = None,
    poll_frequency: float = 10,
) -> tp.Iterator[core.Job[core.R]]:
    """
    Yields jobs as they complete (finished, failed or were cancelled).
    Raises a TimeoutError if the result isn’t available after timeout seconds.
    timeout can be an int or float. If timeout is not specified or None, there is no
    limit to the wait time.

    Parameters
    ----------
    jobs: list
        Jobs instances

    timeout: int/float
        Maximum time (in sec) to wait for jobs completion

    poll_frequency: float
        Frequency in second at which we check job status.

    Yields
    ------
    Job
        The next completed job
    """
    start = time.time()
    jobs_done: tp.Set[int] = set()
    while True:
        if timeout is not None and time.time() - start > timeout:
            raise TimeoutError
        for i, job in enumerate(jobs):
            if i in jobs_done:
                continue
            if job.done():
                jobs_done.add(i)
                yield job
        if len(jobs_done) == len(jobs):
            break
        time.sleep(poll_frequency)


def run_cmd(str_args, **kwargs):
    return subprocess.check_output(str_args, **kwargs).decode("utf-8").strip()


class RsyncSnapshot:
    """Rsync Snapshot
    This class creates a snapshot of the git repository that the script lives in
    when creating the snapshot.  This is useful for ensuring that remote jobs that
    get launched don't accidentally pick up unintended local changes.

    Parameters
    ----------
    snapshot_dir: Path
        A path to where the snapshot should be created
    with_submodules: bool
        Whether or not submodules should be included in the snapshot
    exclude: Sequence[str]
        An optional list of patterns to exclude from the snapshot

    Note
    ----
    - Only files that are checked in to the repository are included in the snapshot.
        If you have experimental code that you would like to include in the snapshot,
        you'll need to `git add` the file first for it to be included
    """

    def __init__(self, snapshot_dir: Path, with_submodules: bool = False, exclude: tp.Sequence[str] = ()):
        if shutil.which("rsync") is None:
            raise RuntimeError("RsyncSnapshot requires rsync to be installed.")
        self.snapshot_dir = Path(snapshot_dir)
        self.original_dir = Path.cwd()
        self.with_submodules = with_submodules
        self.exclude = exclude

    def __enter__(self) -> None:
        self.original_dir = Path.cwd()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Get the repository root
        root_dir = run_cmd(["git", "rev-parse", "--show-toplevel"])
        sub = "--recurse-submodules" if self.with_submodules else "-s"
        # Get a list of all the checked in files that we can pass to rsync
        with tempfile.NamedTemporaryFile() as tfile:
            # https://stackoverflow.com/a/51689219/4876946
            run_cmd(f"git ls-files {sub} | grep -v ^16 | cut -f2- > {tfile.name}", cwd=root_dir, shell=True)
            exclude = list(itertools.chain.from_iterable(("--exclude", pat) for pat in self.exclude))
            run_cmd(["rsync", "-a", "--files-from", tfile.name, root_dir, str(self.snapshot_dir)] + exclude)
        os.chdir(self.snapshot_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)
