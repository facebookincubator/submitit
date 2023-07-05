# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import typing as tp
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..core import core, job_environment, utils


class OarInfoWatcher(core.InfoWatcher):

    submitit_state_mapping = {
        "Suspended": "SUSPENDED",
        "Hold": "PENDING",
        "toLaunch": "READY",
        "Error": "FAILED",
        "toError": "FAILED",
        "toAckReservation": "PENDING",
        "Waiting": "PENDING",
        "Running": "RUNNING",
        "Finishing": "COMPLETING",
        "Terminated": "COMPLETED",
        "Launching": "READY",
        "Resuming": " REQUEUED",
    }

    def _make_command(self) -> Optional[List[str]]:
        # asking for array id will return all status
        # on the other end, asking for each and every one of them individually takes a huge amount of time
        to_check = {x for x in self._registered - self._finished}
        if not to_check:
            return None
        command = ["oarstat", "-f", "-J"]
        for jid in to_check:
            command.extend(["-j", str(jid)])
        return command

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        """Returns the state of the job
        State of finished jobs are cached (use watcher.clear() to remove all cache)

        Parameters
        ----------
        job_id: int
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        info = self.get_info(job_id, mode=mode)
        return info.get("State") or "UNKNOWN"

    def read_info(self, string: Union[bytes, str]) -> Dict[str, Dict[str, str]]:
        """Reads the output of oarstat and returns a dictionary containing main information"""
        all_stats: Dict[str, Dict[str, str]] = {}
        if string:
            if not isinstance(string, str):
                string = string.decode()
            oarstat_output_dict = json.loads(string)
            if len(oarstat_output_dict) == 0:
                return {}  # one job id does not exist (yet)
            for k,v in oarstat_output_dict.items():
                stat = {
                    "JobID": k,
                    "State": self.submitit_state_mapping.get(v.get("state")),
                    "NodeList": v.get("assigned_network_address")}
                all_stats[k] = stat
        return all_stats


class OarJob(core.Job[core.R]):

    _cancel_command = "oardel"
    watcher = OarInfoWatcher(delay_s=600)

    def _interrupt(self, timeout: bool = False) -> None:
        """Sends preemption or timeout signal to the job (for testing purpose)

        Parameter
        ---------
        timeout: bool
            Whether to trigger a job time-out (if False, it triggers preemption)
        """
        # in case of preemption, the job will be checkpointed with signal 12 (SIGUSR2)
        if not timeout:
            subprocess.check_call([self._cancel_command, "-s", "SIGTERM", self.job_id], shell=False)
        subprocess.check_call([self._cancel_command, "-c" , self.job_id], shell=False)


class OarExecutor(core.PicklingExecutor):
    """Oar job executor
    This class is used to hold the parameters to run a job on OAR.
    In practice, it will create a batch file in the specified directory for each job,
    and pickle the task function and parameters. At completion, the job will also pickle
    the output. Logs are also dumped in the same directory.

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.
    max_num_timeout: int
        Maximum number of time the job can be requeued after timeout (if
        the instance is derived from helpers.Checkpointable)

    Note
    ----
    - be aware that the log/output folder will be full of logs and pickled objects very fast,
      it may need cleaning.
    - the folder needs to point to a directory shared through the cluster. This is typically
      not the case for your tmp! If you try to use it, it will fail silently (since it
      will not even be able to log stderr).
    - use update_parameters to specify custom parameters (queue etc...). If you
      input erroneous parameters, an error will print all parameters available for you.
    """

    job_class = OarJob

    def __init__(self, folder: Union[Path, str], max_num_timeout: int = 3) -> None:
        super().__init__(folder, max_num_timeout=max_num_timeout)
        if not self.affinity() > 0:
            raise RuntimeError('Could not detect "oarsub", are you indeed on a OAR cluster?')

    @classmethod
    def _equivalence_dict(cls) -> core.EquivalenceDict:
        return {
            "name": "n",
            "timeout_min": "timeout_min",
            "nodes": "nodes",
            "gpus_per_node": "gpu",
        }

    @classmethod
    def _valid_parameters(cls) -> Set[str]:
        """Parameters that can be set through update_parameters"""
        return set(_get_default_parameters())

    def _internal_update_parameters(self, **kwargs: Any) -> None:
        """Updates oar submission parameters

        Parameters
        ----------
        See oar documentation for most parameters.
        Most useful parameters are: core, walltime, gpu, queue.

        Below are the parameters that differ from OAR documentation:

        folder: str/Path
            folder where print logs and error logs will be written
        additional_parameters: dict
            add OAR parameters which are not currently available in submitit.
            Eg: {"t": ["besteffort", "idempotent"]} will be prepended as "#OAR -t besteffort -t idempotent" in the OAR submition file.
            Eg: {"p": "'chetemi AND memcore>=3337'"} will be prepended as "#OAR -p 'chetemi AND memcore>=3337'" in the OAR submission file.

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values
        """
        defaults = _get_default_parameters()
        in_valid_parameters = sorted(set(kwargs) - set(defaults))
        if in_valid_parameters:
            string = "\n  - ".join(f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items()))
            raise ValueError(
                f"Unavailable parameter(s): {in_valid_parameters}\nValid parameters are:\n  - {string}"
            )
        if 'walltime' in kwargs:
            # use updated OAR specific parameter walltime by default if present
            kwargs['timeout_min'] = _oar_walltime_to_timeout_min(kwargs['walltime'])
        elif 'timeout_min' in kwargs:
            # if shared parameter timeout_min is updated but not the OAR specific parameter walltime,
            # then convert timeout_min in minutes to [hour:min:sec|hour:min|hour] format for OAR.
            kwargs['walltime'] = _timeout_min_to_oar_walltime(kwargs['timeout_min'])
        # check that new parameters are correct
        _make_oarsub_string(command="nothing to do", folder=self.folder, **kwargs)
        super()._internal_update_parameters(**kwargs)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            return super()._internal_process_submissions(delayed_submissions)
        #TODO deal with job array
        raise NotImplementedError

    @property
    def _submitit_command_str(self) -> str:
        return " ".join(
            [shlex.quote(sys.executable), "-u -m submitit.core._submit", shlex.quote(str(self.folder))]
        )

    def _num_tasks(self) -> int:
        return 1

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        return _make_oarsub_string(command=command, folder=self.folder, **self.parameters)

    def _make_submission_command(self, submission_file_path: Path) -> List[str]:
        # For OAR cluster, the submission file needs to exist at the submission time
        # AND at the job launch time.
        # That's why we should override this parameter of _submit_command method from PicklingExecutor.
        # Instead of submitting using the temp submission file,
        # we read from this temp file, and submit using an inline command.
        with open(submission_file_path) as f:
            submission_script_lines = f.readlines()
        oarsub_options = [line[5:].strip() for line in submission_script_lines if line.startswith("#OAR ")]
        oarsub_cmd = " ".join(["oarsub "] + oarsub_options)
        inline_script_lines = [line for line in submission_script_lines if not line.startswith("#")]
        inline_cmd = "".join(inline_script_lines)
        return shlex.split(oarsub_cmd) + [inline_cmd]

    @staticmethod
    def _get_job_id_from_submission_command(string: Union[bytes, str]) -> str:
        """Returns the job ID from the output of oarsub string"""
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"OAR_JOB_ID=(?P<id>[0-9]+)", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of oarsub output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")

    @classmethod
    def affinity(cls) -> int:
        return -1 if shutil.which("oarsub") is None else 2


@functools.lru_cache()
def _get_default_parameters() -> Dict[str, Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_oarsub_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults) # type: ignore
    return {key: val for key, val in zipped if key not in {"command", "folder", "map_count"}}


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_oarsub_string(
    command: str,
    folder: tp.Union[str, Path],
    nodes: tp.Optional[int] = None,
    core: tp.Optional[int] = None,
    gpu: tp.Optional[int] = None,
    walltime: tp.Optional[str] = None,
    timeout_min: tp.Optional[int] = None,
    queue: tp.Optional[str] = None,
    n: str = "submitit",
    additional_parameters: tp.Optional[tp.Dict[str, tp.Union[List[str], str]]] = None,
) -> str:
    """Creates the content of a bash file with provided parameters

    Parameters
    ----------
    See oar documentation for most parameters.
    Most useful parameters are: core, walltime, gpu, queue.

    Below are the parameters that differ from OAR documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    additional_parameters: dict
        add OAR parameters which are not currently available in submitit.
        Eg: {"t": ["besteffort", "idempotent"]} will be prepended as "#OAR -t besteffort -t idempotent" in the OAR submition file.
        Eg: {"p": "'chetemi AND memcore>=3337'"} will be prepended as "#OAR -p 'chetemi AND memcore>=3337'" in the OAR submission file.

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    parameters = {}
    # resource request passed to the "-l" option
    # OAR needs to have the resource request on a single line, otherwise it is considered as a moldable job.
    # OAR resource hierarchy: nodes > gpu > core
    resource_hierarchy = ""
    if nodes is not None:
        resource_hierarchy = "/nodes=%d" % nodes
    if gpu is not None:
        resource_hierarchy += "/gpu=%d" % gpu
    if core is not None:
        resource_hierarchy +=  "/core=%d" % core
    if walltime is not None:
        walltime = "walltime=%s" % walltime
    resource_request = ",".join(filter(None, (resource_hierarchy, walltime)))
    if resource_request:
        parameters["l"] = resource_request
    # queue parameter passed to the "-q" option
    if queue is not None:
        parameters["q"] = queue
    # name parameter passed to the "-n" option
    parameters["n"] = n
    # stdout and stderr passed to OAR "-O" and "-E" options
    paths = utils.JobPaths(folder=folder)
    parameters["O"] = str(paths.stdout).replace("%j", "%jobid%").replace("%t", "0")
    parameters["E"] = str(paths.stderr).replace("%j", "%jobid%").replace("%t", "0")
    # additional parameters passed here
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create the bash file with OAR options
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(_as_oar_flag(k, parameters[k]))
    lines += ["", "# command", "export SUBMITIT_EXECUTOR=oar", command, ""]
    return "\n".join(lines)


def _as_oar_flag(key: str, value: tp.Any) -> str:
    key = key.replace("_", "-")
    if isinstance(value, list):
        values = " ".join(f"-{key} {v}" for v in value)
        return f"#OAR {values}"
    elif isinstance(value, str):
        return f"#OAR -{key} {value}"
    else:
        return f"#OAR -{key} {str(value)}"


def _timeout_min_to_oar_walltime(timeout_min: int) -> str:
    hour = timeout_min // 60
    minute = timeout_min - hour * 60
    return f"{hour:02}:{minute:02}"


def _oar_walltime_to_timeout_min(walltime: str) -> int:
    # Split the walltime string in [hour:min:sec|hour:min|hour] format
    parts = walltime.split(':')
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    seconds = int(parts[2]) if len(parts) > 2 else 0
    return hours * 60 + minutes + round(seconds / 60)


class OarJobEnvironment(job_environment.JobEnvironment):

    _env = {
        "job_id": "OAR_JOB_ID",
        "nodes": "OAR_NODEFILE",
        "array_task_id": "OAR_ARRAY_INDEX",
        "num_tasks": "",
        "local_rank": "",
        "global_rank": "",
    }

    @property
    def hostnames(self) -> List[str]:
        # Parse the content of the "OAR_NODEFILE" environment variable,
        # which gives access to the list of hostnames that are part of the current job.
        nodes_file_path = os.environ.get(self._env["nodes"], "")
        if os.path.exists(nodes_file_path):
            with open(nodes_file_path) as f:
                # read lines and remove duplicates
                node_list = list(set(f.readlines()))
            # remove the end of line "\n" for hostnames list
            # and sort the hostnames list in alphabetical order
            return sorted([n.strip() for n in node_list])
        else:
            return [self.hostname]

    @property
    def num_nodes(self) -> int:
        """Total number of nodes for the job:"""
        # For OAR, the "num_nodes" environment variable does not exist.
        # We count the number of hostnames for the current job.
        if not self.hostnames:
            return 1
        return len(self.hostnames)

    @property
    def node(self) -> int:
        """Id of the current node:"""
        # For OAR, the "node" environment variable does not exist.
        # We sort the hostnames of one job in alphabetical order,
        # and then return the index of hostname in the hostnames list.
        if not self.hostnames or self.hostname not in self.hostnames:
            return 0
        return self.hostnames.index(self.hostname)
