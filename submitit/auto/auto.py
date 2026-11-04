# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import typing as tp
import warnings
from pathlib import Path
from typing import Any

from ..core import plugins
from ..core.core import Executor, Job
from ..core.utils import DelayedSubmission


def _convert_deprecated_args(kwargs: dict[str, Any], deprecated_args: tp.Mapping[str, str]) -> None:
    for arg in list(kwargs):
        new_arg = deprecated_args.get(arg)
        if not new_arg:
            continue
        kwargs[new_arg] = kwargs.pop(arg)
        warnings.warn(f"Setting '{arg}' is deprecated. Use '{new_arg}' instead.")


class AutoExecutor(Executor):
    """Executor submitting to the best available cluster: Slurm when ``srun`` is available,
    the local executor otherwise.
    Prefer it to the other executors, so that the same code runs on several clusters.

    Parameters
    ----------
    folder: Path | str
        folder for storing job submission/output and logs, ``%j`` is replaced by the job id.
    cluster: str
        executor name: "slurm", "local" (even on a Slurm cluster), "debug" or a plugin's
        (default: detected, see ``executor.cluster``)
    kwargs:
        executor constructor arguments, prefixed by the executor name, e.g. ``slurm_python``
        for the ``python`` argument of :class:`~submitit.SlurmExecutor`.

    Update parameters
    -----------------
    name: str
        job name
    timeout_min: int
        time limit in minutes (defaults: see each executor)
    mem_gb: float
        memory per node in GB
    nodes: int
        number of nodes
    tasks_per_node: int
        number of tasks per node, each running the submitted function
    cpus_per_task: int
        number of CPUs per task
    gpus_per_node: int
        number of GPUs per node
    stderr_to_stdout: bool
        writes stderr to the stdout log
    <executor>_<parameter>:
        parameter of one executor, e.g. ``slurm_partition`` (see :class:`~submitit.SlurmExecutor`
        and :class:`~submitit.LocalExecutor`), ignored by the other executors.
        A prefixed generic parameter, e.g. ``slurm_timeout_min``, overrides the generic one on
        that executor only.
    """

    _ctor_deprecated_args = {"max_num_timeout": "slurm_max_num_timeout", "conda_file": "chronos_conda_file"}

    def __init__(self, folder: str | Path, cluster: str | None = None, **kwargs: Any) -> None:
        self.cluster = cluster or self.which()

        executors = plugins.get_executors()
        if self.cluster not in executors:
            raise ValueError(f"AutoExecutor doesn't know any executor named {self.cluster}")

        _convert_deprecated_args(kwargs, self._ctor_deprecated_args)
        err = "Extra arguments must be prefixed by executor named, received unknown arg"
        err_ex_list = f"Known executors: {', '.join(executors)}."
        for name in kwargs:
            assert "_" in name, f"{err} '{name}'. {err_ex_list}"
            prefix = name.split("_")[0]
            assert prefix in executors, (
                f"{err} '{name}', and '{prefix}' executor is also unknown. {err_ex_list}"
            )
        self._executor = flexible_init(executors[self.cluster], folder, **kwargs)

        valid = self._valid_parameters()
        self._deprecated_args = {
            arg: f"{ex_name}_{arg}"
            for ex_name, ex in sorted(
                executors.items(), key=lambda item: item[0] == self.cluster
            )  # current wins
            for arg in ex._valid_parameters()
            if arg not in valid
        }
        super().__init__(self._executor.folder, self._executor.parameters)

    @staticmethod
    def which() -> str:
        """Returns what is the detected cluster."""
        executors = plugins.get_executors()
        best_ex = max(executors, key=lambda ex: executors[ex].affinity())

        if executors[best_ex].affinity() <= 0:
            raise RuntimeError(f"Did not found an available executor among {executors.keys()}.")

        return best_ex

    def register_dev_folders(self, folders: list[str | Path]) -> None:
        """Archive a list of folders to be untarred in the job working directory.
        This is only implemented for internal cluster, for running job on non-installed packages.
        This is not useful on slurm since the working directory of jobs is identical to
        your work station working directory.

        folders: list of paths
            The list of folders to archive and untar in the job working directory
        """
        register = getattr(self._executor, "register_dev_folders", None)
        if register is not None:
            register(folders)
        else:
            # TODO this should be done through update parameters
            warnings.warn(
                "Ignoring dev folder registration as it is only supported (and needed) for internal cluster"
            )

    @classmethod
    def _typed_parameters(cls) -> dict[str, type]:
        return {
            "name": str,
            "timeout_min": int,
            "mem_gb": float,
            "nodes": int,
            "cpus_per_task": int,
            "gpus_per_node": int,
            "tasks_per_node": int,
            "stderr_to_stdout": bool,
        }

    @classmethod
    def _valid_parameters(cls) -> set[str]:
        return set(cls._typed_parameters().keys())

    def _internal_update_parameters(self, **kwargs: Any) -> None:
        """Updates submission parameters to srun/crun.

        Parameters
        ----------
        AutoExecutors provides shared parameters that are translated for each specific cluster.
        Those are: timeout_min (int), mem_gb (int), gpus_per_node (int), cpus_per_task (int),
        nodes (int), tasks_per_node (int) and name (str).
        Cluster specific parameters can be specified by prefixing them with the cluster name.

        Notes
        -----
        - Cluster specific parameters win over shared parameters.
            eg: if both `slurm_time` and `timeout_min` are provided, then:
                - `slurm_time` is used on the slurm cluster
                - `timeout_min` is used on other clusters
        """
        # We handle None as not set.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # check type of replaced variables
        generics = AutoExecutor._typed_parameters()
        for name, expected_type in generics.items():
            if expected_type == float:
                expected_type = (int, float)  # type: ignore
            if name in kwargs:
                assert isinstance(kwargs[name], expected_type), (
                    f'Parameter "{name}" expected type {expected_type} (but value: "{kwargs[name]}")'
                )

        _convert_deprecated_args(kwargs, self._deprecated_args)
        specific = [x.split("_", 1) for x in kwargs if x not in generics]

        invalid = []
        executors = plugins.get_executors()
        for ex_arg in specific:
            if len(ex_arg) != 2:
                invalid.append(f"Parameter '{ex_arg[0]}' need to be prefixed by an executor name.")
                continue
            ex, arg = ex_arg

            if ex not in executors:
                invalid.append(f"Unknown executor '{ex}' in parameter '{ex}_{arg}'.")
                continue

            valid = executors[ex]._valid_parameters()
            if arg not in valid and arg not in generics:
                invalid.append(
                    f"Unknown argument '{arg}' for executor '{ex}' in parameter '{ex}_{arg}'."
                    + " Valid arguments: "
                    + ", ".join(valid)
                )
                continue
        if invalid:
            invalid.extend(
                [
                    f"Known executors: {', '.join(executors.keys())}",
                    f"As a reminder, shared/generic (non-prefixed) parameters are: {generics}.",
                ]
            )
            raise NameError("\n".join(invalid))

        # add cluster specific generic overrides
        kwargs.update(
            **{
                arg: kwargs.pop(f"{ex}_{arg}")
                for ex, arg in specific
                if ex == self.cluster and arg in generics
            }
        )
        parameters = self._executor._convert_parameters({k: kwargs[k] for k in kwargs if k in generics})
        # update parameters in the core executor
        for ex, arg in specific:
            # update cluster specific non-generic arguments
            if arg not in generics and ex == self.cluster:
                parameters[arg] = kwargs[f"{ex}_{arg}"]

        self._executor._internal_update_parameters(**parameters)

    def _internal_process_submissions(
        self, delayed_submissions: list[DelayedSubmission]
    ) -> list[Job[tp.Any]]:
        return self._executor._internal_process_submissions(delayed_submissions)


def flexible_init(cls: type[Executor], folder: str | Path, **kwargs: Any) -> Executor:
    prefix = cls.name() + "_"
    return cls(folder, **{k.removeprefix(prefix): val for k, val in kwargs.items() if k.startswith(prefix)})
