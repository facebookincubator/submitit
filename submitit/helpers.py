# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import collections
import datetime
import itertools
import os
import random
import shutil
import subprocess
import tempfile
import time
import typing as tp
from pathlib import Path

# pylint: disable=unused-import
# import DelayedSubmission and CommandFunction to populate helpers namespace
from .core import core
from .core.job_environment import JobEnvironment
from .core.utils import CommandFunction as CommandFunction  # noqa
from .core.utils import DelayedSubmission as DelayedSubmission  # noqa
from .core.utils import environment_variables as environment_variables  # noqa


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
    """Takes a snapshot of the git repository that the script lives in.

    This ensures that remote jobs always use the code from when they are scheduled
    and not the code from when they are launched / re-started.


    Parameters
    ----------
    snapshot_dir: Path
        A path to where the snapshot should be created
    with_submodules: bool
        Whether or not submodules should be included in the snapshot
    exclude: Sequence[str]
        An optional list of patterns to exclude from the snapshot
    include: Sequence[str]
        A list of relative file names to include from the snapshot.
        Useful for .so or other build artifacts that are genarally not tracked by git.

    Note
    ----
    - Only files that are checked in to the repository are included in the snapshot.
        If you have experimental code that you would like to include in the snapshot,
        you'll need to `git add` the file first for it to be included, or use `include` arg.
    """

    def __init__(
        self,
        snapshot_dir: Path,
        root_dir: Path = None,
        with_submodules: bool = False,
        exclude: tp.Sequence[str] = (),
        include: tp.Sequence[str] = (),
    ):
        self.available(throw=True)
        self.snapshot_dir = Path(snapshot_dir)
        self.root_dir = root_dir or run_cmd(["git", "rev-parse", "--show-toplevel"])
        self.original_dir = Path.cwd()
        self.with_submodules = with_submodules
        self.exclude = exclude
        self.include = include

    @staticmethod
    def available(throw: bool = False) -> bool:
        if not shutil.which("rsync"):
            if throw:
                raise RuntimeError("RsyncSnapshot requires rsync to be installed.")
            return False
        return True

    def __enter__(self) -> None:
        self.original_dir = Path.cwd()
        # Get the repository root
        root_dir = str(self.root_dir)
        sub = "--recurse-submodules" if self.with_submodules else "-s"
        # Make a shallow git clone
        if not self.snapshot_dir.exists():
            self.snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["git", "clone", "--depth=2", f"file://{root_dir}", str(self.snapshot_dir)])

        # Get a list of all the checked in files that we can pass to rsync
        # Is Rsync faster than a `git pull` ?
        with tempfile.NamedTemporaryFile() as tfile:
            # https://stackoverflow.com/a/51689219/4876946
            run_cmd(f"git ls-files {sub} | grep -v ^16 | cut -f2- > {tfile.name}", cwd=root_dir, shell=True)
            exclude = list(itertools.chain.from_iterable(("--exclude", pat) for pat in self.exclude))
            with open(tfile.name, "a", encoding="utf8") as o:
                for inc in self.include:
                    print(inc, file=o)
            run_cmd(["rsync", "-a", "--files-from", tfile.name, root_dir, str(self.snapshot_dir)] + exclude)
        os.chdir(self.snapshot_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)


def _default_custom_logging(monitoring_start_time: float, n_jobs: int, state_jobs: tp.Dict[str, tp.Set[int]]):
    run_time = time.time() - monitoring_start_time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failed_job_indices = sorted(state_jobs["FAILED"])
    n_chars = len(str(n_jobs))

    print(
        f"[{date_time}] Launched {int(run_time / 60)} minutes ago,",
        f"{len(state_jobs['RUNNING']):{n_chars}}/{n_jobs} jobs running,",
        f"{len(failed_job_indices):{n_chars}}/{n_jobs} jobs failed,",
        f"{len(state_jobs['DONE']) - len(failed_job_indices):{n_chars}}/{n_jobs} jobs done",
        flush=True,
    )

    if len(failed_job_indices) > 0:
        print(f"[{date_time}] Failed jobs, indices {failed_job_indices}", flush=True)


def monitor_jobs(
    jobs: tp.Sequence[core.Job[core.R]],
    poll_frequency: float = 30,
    test_mode: bool = False,
    custom_logging: tp.Callable = _default_custom_logging,
) -> None:
    """Continuously monitors given jobs until they are all done or failed.

    Parameters
    ----------
    jobs: List[Jobs]
        A list of jobs to monitor
    poll_frequency: int
        The time (in seconds) between two refreshes of the monitoring.
        Can't be inferior to 30s.
    test_mode: bool
        If in test mode, we do not check the length of poll_frequency
    """

    if not test_mode:
        assert poll_frequency >= 30, "You can't refresh too often (>= 30s) to avoid overloading squeue"

    n_jobs = len(jobs)
    if n_jobs == 0:
        print("There are no jobs to monitor")
        return

    job_arrays = ", ".join(sorted(set(str(job.job_id).split("_", 1)[0] for job in jobs)))
    print(f"Monitoring {n_jobs} jobs from job arrays {job_arrays} \n")

    monitoring_start_time = time.time()
    while True:
        if not test_mode:
            jobs[0].get_info(mode="force")  # Force update once to sync the state
        state_jobs = collections.defaultdict(set)
        for i, job in enumerate(jobs):
            state_jobs[job.state.upper()].add(i)
            if job.done():
                state_jobs["DONE"].add(i)

        failed_job_indices = sorted(state_jobs["FAILED"])
        if len(state_jobs["DONE"]) == len(jobs):
            print(f"All jobs finished, jobs with indices {failed_job_indices} failed", flush=True)
            break

        custom_logging(monitoring_start_time, n_jobs, state_jobs)
        time.sleep(poll_frequency)

    print(f"Whole process is finished, took {int((time.time() - monitoring_start_time) / 60)} minutes")


class TorchDistributedParams(tp.NamedTuple):
    master_addr: str
    master_port: int
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int


def export_torch_distributed_env() -> TorchDistributedParams:
    """Export the required environment variables to initialize PyTorch distributed (with the default env:// method).

    Returns
    -------
    params: TorchDistributedParams
        a named tuple with the master node address and port and the assigned rank, world size, local rank and local world size.
    """
    #MIN_MASTER_PORT, MAX_MASTER_PORT = (1023, 65535)
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
    master_port_str = os.environ("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(job_env.job_id)
        master_port = rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)
    else:
        master_port = int(master_port_str)
        assert master_port >= MIN_MASTER_PORT
        assert master_port <= MAX_MASTER_PORT

    # See the "Environment variable initialization" section from
    # https://pytorch.org/docs/stable/distributed.html for the complete list of
    # environment variables required for the env:// initialization method.
    job_env = JobEnvironment()
    params = TorchDistributedParams(  # pylint: disable=no-value-for-parameter
        master_addr=job_env.hostnames[0],
        master_port=master_port,
        rank=job_env.global_rank,
        world_size=job_env.num_tasks,
        local_rank=job_env.local_rank,
        local_world_size=job_env.num_tasks // job_env.num_nodes,
    )
    env_vars = {
        "MASTER_ADDR": params.master_addr,
        "MASTER_PORT": str(params.master_port),
        "RANK": str(params.rank),
        "WORLD_SIZE": str(params.world_size),
        "LOCAL_RANK": str(params.local_rank),  # Not required
        "LOCAL_WORLD_SIZE": str(params.local_world_size),  # Not required
    }
    for key in env_vars:
        assert os.environ.get(key) is None
    os.environ.update(env_vars)
    return params
