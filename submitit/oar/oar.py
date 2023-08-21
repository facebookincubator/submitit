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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .. import helpers
from ..core import core, job_environment, logger, utils


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

    def __init__(self, folder: tp.Union[Path, str], job_id: str, tasks: tp.Sequence[int] = (0,)) -> None:
        if len(tasks) > 1:
            raise NotImplementedError
        super().__init__(folder, job_id, tasks)
        self._resubmitted_job = None

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

    def cancel(self, check: bool = True) -> None:
        """Cancels the job

        Parameters
        ----------
        check: bool
            whether to wait for completion and check that the command worked
        """
        # The "oardel --array job_id" command will cancel the job, which means on OAR cluster:
        # the original job and the resubmitted jobs, represented by the original job's job_id in case of checkpointing,
        # all jobs represented by the first job's job_id in case of job array,
        # or a basic job.
        (subprocess.check_call if check else subprocess.call)(
            self._get_cancel_command() + [self.job_id], shell=False
        )

    def _get_cancel_command(self) -> List[str]:
        if self._resubmitted_job is None:
            return [self._cancel_command]
        else:
            # "--array" ensure to delete the original job and the resubmitted ones
            return [self._cancel_command, '--array']

    def done(self, force_check: bool = False) -> bool:
        """Checks whether the job is finished.
        This is done by checking if the result file is present,
        or checking the job state regularly (at least every minute)

        Parameters
        ----------
        force_check: bool
            Forces the OAR state update

        Returns
        -------
        bool
            whether the job is finished or not

        Note
        ----
        This function is not foolproof, and may say that the job is not terminated even
        if it is when the job failed (no result file, but job not running) because
        we avoid calling oarstat everytime done is called
        """
        # If the job is checkpointed and resubmitted,
        # the job is done once the original job and the resubmitted one are all done.
        if self._resubmitted_job is None:
            if super().done(force_check):
                if self._get_resubmitted_job() is None:
                    return True
            else:
                return False
        return self._resubmitted_job.done(force_check)

    def _get_resubmitted_job(self) -> "OarJob[core.R]":
        """Returns the resubmitted job.
        If the job is not resubmitted, return None
        """
        if self._resubmitted_job is None:
            command = ["oarstat", "--sql", f"resubmit_job_id='{self.job_id}'", "-J"]
            try:
                logger.get_logger().debug(f"Call command {' '.join(command)}")
                output = subprocess.check_output(command, shell=False)
                output_dict = self.watcher.read_info(output)
                # resubmitted_job_id is the key of the output_dict
                resubmitted_job_id = next(iter(output_dict.keys()), None)
                if resubmitted_job_id is not None:
                    self._resubmitted_job = OarJob(folder=self._paths.folder, job_id=resubmitted_job_id, tasks=[0])
            except Exception as e:
                logger.get_logger().error(f"Getting error with _get_resubmitted_job() by command {command}:\n")
                raise e
        return self._resubmitted_job

    def _get_logs_string(self, name: str) -> tp.Optional[str]:
        """Returns a string with the content of the log files
        or None if the file does not exist yet

        Parameter
        ---------
        name: str
            either "stdout" or "stderr"
        """
        string = super()._get_logs_string(name)
        if self._get_resubmitted_job() is not None:
            string += self._resubmitted_job._get_logs_string(name)
        return string


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
        Maximum number of time the job can be resubmitted after timeout (if
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
    watcher = OarInfoWatcher(delay_s=600)

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

        setup: list
            a list of command to run in sbatch before running srun
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

    def _get_checkpointable_executor(self):
        ex = OarExecutor(self.folder, self.max_num_timeout)
        ex.update_parameters(**self.parameters)
        # set the OAR Job type to idempotent,
        # in this way, the checkpointed job will be resubmitted automatically by OAR.
        ex.parameters.setdefault('additional_parameters', {})
        ex.parameters['additional_parameters'].setdefault('t', [])
        ex.parameters['additional_parameters']['t'].append('idempotent')
        return ex

    def _need_checkpointable_executor(self, delayed_submission: utils.DelayedSubmission):
        return isinstance(delayed_submission.function, helpers.Checkpointable) and \
                ('additional_parameters' not in self.parameters or \
                't' not in self.parameters['additional_parameters'] or \
                'idempotent' not in self.parameters['additional_parameters']['t'])

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            if self._need_checkpointable_executor(delayed_submissions[0]):
                # Make a copy of the executor, since we don't want other jobs to be
                # scheduled with idempotent type
                executor = self._get_checkpointable_executor()
                return executor._internal_process_submissions(delayed_submissions)
            return super()._internal_process_submissions(delayed_submissions)
        # array
        # delayed_submissions should be either all Checkpointable functions or all non Checkpointable functions
        if any(isinstance(d.function, helpers.Checkpointable) for d in delayed_submissions) and \
            any(not isinstance(d.function, helpers.Checkpointable) for d in delayed_submissions):
            raise Exception("OarExecutor does not support a job array that mixes checkpointable and non-checkpointable functions."
                            "\nPlease make groups of similar function calls in the job array.")
        folder = utils.JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("timeout_min", 5)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        if self._need_checkpointable_executor(delayed_submissions[0]):
            array_ex = self._get_checkpointable_executor()
        else:
            array_ex = OarExecutor(self.folder, self.max_num_timeout)
            array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[tp.Any] = array_ex._submit_command(self._submitit_command_str)
        jobIdList = self._get_job_id_list_from_array_id(first_job.job_id)
        jobs: List[core.Job[tp.Any]] = [
            OarJob(folder=self.folder, job_id=f"{jid}", tasks=[0]) for jid in jobIdList
        ] # only single task is supported for the moment.
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs

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
        oarsub_options = [item for line in submission_script_lines if line.startswith("#OAR ") for item in line[5:].split()]
        inline_script_lines = [line.strip() for line in submission_script_lines if not line.startswith("#") and line != '\n']
        return ["oarsub"] + oarsub_options + [ "; ".join(inline_script_lines) ]

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

    def _get_job_id_list_from_array_id(self, array_id) -> [str]:
        """Returns the list of OAR jobid of a job array"""
        command = ["oarstat", "--array", array_id, "-J"]
        try:
            logger.get_logger().debug(f"Call command {' '.join(command)}")
            output = subprocess.check_output(command, shell=False)
            output_dict = self.watcher.read_info(output)
            return sorted(list(output_dict.keys()))
        except Exception as e:
            logger.get_logger().error(f"Getting error with _get_job_id_list_from_array_id() by command {command}:\n")
            raise e


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
    map_count: tp.Optional[int] = None,  # used internally
    nodes: tp.Optional[int] = None,
    core: tp.Optional[int] = None,
    gpu: tp.Optional[int] = None,
    walltime: tp.Optional[str] = None,
    timeout_min: tp.Optional[int] = None,
    queue: tp.Optional[str] = None,
    setup: tp.Optional[tp.List[str]] = None,
    n: str = "submitit",
    stderr_to_stdout: bool = False,
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
    setup: list
        a list of command to run in sbatch befure running srun
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
        resource_hierarchy += "/nodes=%d" % nodes
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
    parameters["E"] = parameters["O"] if stderr_to_stdout else str(paths.stderr).replace("%j", "%jobid%").replace("%t", "0")
    if map_count is not None:
        assert isinstance(map_count, int)
        parameters["array"] = map_count
    # additional parameters passed here
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create the bash file with OAR options
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(_as_oar_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup
    lines += ["", "# command", "export SUBMITIT_EXECUTOR=oar", command, ""]
    return "\n".join(lines)


def _as_oar_flag(key: str, value: tp.Any) -> str:
    if len(key) == 1:
        key = f"-{key}"
    else:
        key = f"--{key}"
    if isinstance(value, list):
        values = " ".join(f"{key} {v}" for v in value)
        return f"#OAR {values}"
    else:
        return f"#OAR {key} {value}"


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
        "array_job_id": "OAR_ARRAY_ID",
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
    def array_task_id(self) -> Optional[str]:
        if os.environ.get(self._env["array_task_id"]):
            # For OAR, OAR_ARRAY_INDEX starts from 1 (not 0) initially
            return str(int(os.environ.get(self._env["array_task_id"])) -1)
        return None

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

    def _requeue(self, countdown: int) -> None:
        """Requeue the current job."""
        # Submitit requires _requeue to be overridden by plugin's JobEnvironment implementations.
        # However, OAR does not use requeue mechanism for checkpointing, since a checkpointed job will systematically be terminated by OAR.
        # Instead, we rely on OAR's automatic resubmission mechanism by adding idempotent type on the initial job.
        # Note that only >60s + exit code 99 + idempotent jobs can be resubmitted automatically by OAR.
        logger.get_logger().info(f"Exiting job {self.job_id} with 99 code, ({countdown} remaining timeouts)")
        sys.exit(99)

    @property
    def paths(self) -> utils.JobPaths:
        """Provides the paths used by submitit, including
        stdout, stderr, submitted_pickle and folder.
        """
        folder = os.environ["SUBMITIT_FOLDER"]
        if self.raw_job_id != self.array_job_id and self.array_task_id == "0":
            # Since a resubmitted job is considered as a continuation of the orginal job in a job array,
            # the array_job_id represents also the resubmit_job_id for a resubmitted job.
            # Use here the original job's submitted pickle path instead for signal handling and checkpointing.
            job_id = self.array_job_id
        else:
            # for other jobs, nothing changed, use the raw OAR job's submitted pickle path
            job_id = self.raw_job_id
        return utils.JobPaths(folder, job_id=job_id, task_id=self.global_rank)
