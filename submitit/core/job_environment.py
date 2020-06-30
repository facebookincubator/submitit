# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import signal
import socket
import sys
import types
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence

from . import logger, utils
from .utils import DelayedSubmission, JobPaths


class JobEnvironment:
    """Describe the environment inside which the job is running.
    This includes job id, number of GPUs available, ...

    This class can only be instantiated from a running submitit job.

    @plugin-dev: default implementation look for information into environment variables.
    Override _env to map environment variable to each property.
    """

    _env: ClassVar[Dict[str, str]] = {}

    def __new__(cls, *args: Any) -> "JobEnvironment":
        if cls is not JobEnvironment:
            return super().__new__(cls, *args)  # type: ignore

        from . import plugins  # pylint: disable=cyclic-import,import-outside-toplevel

        return plugins.get_job_environment()

    def __init__(self) -> None:
        self.cluster = self.name()

    @classmethod
    def name(cls) -> str:
        n = cls.__name__
        if n.endswith("JobEnvironment"):
            n = n[: -len("JobEnvironment")]
        return n.lower()

    def activated(self) -> bool:
        """Tests if we are running inside this environment.

        @plugin-dev: assumes that the SUBMITIT_EXECUTOR variable has been
        set to the executor name
        """
        return os.environ.get("SUBMITIT_EXECUTOR", "") == self.name()

    @property
    def hostname(self) -> str:
        return socket.gethostname()

    @property
    def hostnames(self) -> Sequence[str]:
        return [self.hostname]

    @property
    def job_id(self) -> str:
        if self.array_job_id:
            return f"{self.array_job_id}_{self.array_task_id}"
        else:
            return self.raw_job_id

    @property
    def raw_job_id(self) -> str:
        return os.environ[self._env["job_id"]]

    @property
    def array_job_id(self) -> Optional[str]:
        n = "array_job_id"
        return None if n not in self._env else os.environ.get(self._env[n], None)

    @property
    def array_task_id(self) -> Optional[str]:
        n = "array_task_id"
        return None if n not in self._env else os.environ.get(self._env[n], None)

    @property
    def num_tasks(self) -> int:
        """Total number of tasks for the job
        """
        return int(os.environ.get(self._env["num_tasks"], 1))

    @property
    def num_nodes(self) -> int:
        """Total number of nodes for the job
        """
        return int(os.environ.get(self._env["num_nodes"], 1))

    @property
    def node(self) -> int:
        """Id of the current node
        """
        return int(os.environ.get(self._env["node"], 0))

    @property
    def global_rank(self) -> int:
        """Global rank of the task
        """
        return int(os.environ.get(self._env["global_rank"], 0))

    @property
    def local_rank(self) -> int:
        """Local rank of the task, ie on the current node.
        """
        return int(os.environ.get(self._env["local_rank"], 0))

    def __repr__(self) -> str:
        # should look like this:
        # JobEnvironment(job_id=17015819, hostname=learnfair0218, local_rank=2(3), node=1(2), global_rank=5(6))
        info = [f"{n}={getattr(self, n)}" for n in ("job_id", "hostname")]
        names = ("local_rank", "node", "global_rank")
        totals = [self.num_tasks // self.num_nodes, self.num_nodes, self.num_tasks]
        info += [f"{n}={getattr(self, n)}({t})" for n, t in zip(names, totals)]
        return "JobEnvironment({})".format(", ".join(info))

    def _handle_signals(self, paths: JobPaths, submission: DelayedSubmission) -> None:
        """Set up signals handler for the current executable.

        The default implementation checkpoint the given submission and requeues it.
        @plugin-dev: Should be adapted to the signals used in this cluster.
        """
        handler = SignalHandler(self, paths, submission)
        signal.signal(signal.SIGUSR1, handler.checkpoint_and_try_requeue)
        signal.signal(signal.SIGTERM, handler.bypass)

    # pylint: disable=no-self-use,unused-argument
    def _requeue(self, countdown: int) -> None:
        """Requeue the current job.

        @plugin-dev:Must be overridden by JobEnvironment implementations.
            Use self.job_id to find what need to be requeued.
        """
        ...


class SignalHandler:
    def __init__(self, env: JobEnvironment, job_paths: JobPaths, delayed: DelayedSubmission) -> None:
        self.env = env
        self._job_paths = job_paths
        self._delayed = delayed
        self._timedout = True
        self._logger = logger.get_logger()

    def bypass(
        self, signum: signal.Signals, frame: types.FrameType = None  # pylint:disable=unused-argument
    ) -> None:
        self._logger.warning(f"Bypassing signal {signum}")
        self._timedout = False  # this signal before USR1 means the job was preempted

    def checkpoint_and_try_requeue(
        self, signum: signal.Signals, frame: types.FrameType = None  # pylint:disable=unused-argument
    ) -> None:
        case = "timed-out" if self._timedout else "preempted"
        self._logger.warning(f"Caught signal {signum} on {socket.gethostname()}: this job is {case}.")

        procid = self.env.global_rank
        if procid != 0:
            self._logger.info(f"Not checkpointing nor requeuing since I am a slave (procid={procid}).")
            # do not sys.exit, because it might kill the master task
            return

        delayed = self._delayed
        countdown = delayed.timeout_countdown - self._timedout
        no_requeue_reason = ""
        if hasattr(delayed.function, "checkpoint"):
            no_requeue_reason = _checkpoint(self._delayed, self._job_paths.submitted_pickle, countdown)
        elif self._timedout:
            no_requeue_reason = "timed-out and not checkpointable"
        if countdown < 0:  # this is the end
            no_requeue_reason = "timed-out too many times"
        if no_requeue_reason:
            # raise an error so as to create "result_pickle" file which notifies the job is over
            # this is caught by the try/except in "process_job"
            message = f"Job not requeued because: {no_requeue_reason}."
            self._logger.info(message)
            raise utils.UncompletedJobError(message)
        # if everything went well, requeue!
        self.env._requeue(countdown)
        self._exit()

    def checkpoint_and_exit(
        self, signum: signal.Signals, frame: types.FrameType = None  # pylint:disable=unused-argument
    ) -> None:
        self._logger.info(f"Caught signal {signal.Signals(signum).name} on {socket.gethostname()}")

        procid = self.env.global_rank
        if procid:
            self._logger.info(f"Not checkpointing since I am a slave (procid={procid}).")
            # do not sys.exit, because it might kill the master task
            return

        delayed = self._delayed
        if hasattr(delayed.function, "checkpoint"):
            _checkpoint(self._delayed, self._job_paths.submitted_pickle, self._delayed.timeout_countdown)
        self._exit()

    def _exit(self) -> None:
        # extracted for mocking
        self._logger.info("Exiting gracefully after preemption/timeout.")
        sys.exit(-1)


def _checkpoint(delayed: DelayedSubmission, filepath: Path, countdown: int) -> str:
    """Call the checkpoint method and dump the updated delayed.

    Returns:
    --------
        no_requeue_reason: str
            a string explaining while there was no requeuing, else empty string if requeuing works
    """
    logger.get_logger().info("Calling checkpoint method.")
    ckpt_delayed = delayed._checkpoint_function()
    if ckpt_delayed is None:
        return "checkpoint function returned None"
    ckpt_delayed.timeout_countdown = countdown
    with utils.temporary_save_path(filepath) as tmp:
        ckpt_delayed.dump(tmp)
    return ""  # requeues
