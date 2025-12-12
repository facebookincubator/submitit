# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import inspect
import os
import re
import shlex
import shutil
import subprocess
import sys
import typing as tp
import uuid
import warnings
from pathlib import Path

from ..core import core, job_environment, logger, utils


def read_job_id(job_id: str) -> tp.List[tp.Tuple[str, ...]]:
    """Reads formated job id and returns a tuple with format:
    (main_id, [array_index, [final_array_index])
    """
    pattern = r"(?P<main_id>\d+)_\[(?P<arrays>(\d+(-\d+)?(,)?)+)(\%\d+)?\]"
    match = re.search(pattern, job_id)
    if match is not None:
        main = match.group("main_id")
        array_ranges = match.group("arrays").split(",")
        return [tuple([main] + array_range.split("-")) for array_range in array_ranges]
    else:
        main_id, *array_id = job_id.split("_", 1)
        if not array_id:
            return [(main_id,)]
        # there is an array
        array_num = str(int(array_id[0]))  # trying to cast to int to make sure we understand
        return [(main_id, array_num)]


class PBSInfoWatcher(core.InfoWatcher):
    def _make_command(self) -> tp.Optional[tp.List[str]]:
        # ask qstat for full info on each main job id
        to_check = {x.split("_")[0] for x in self._registered - self._finished}
        if not to_check:
            return None
        # qstat -f <jobid> returns full job info for each requested id
        return ["qstat", "-f", *map(str, to_check)]

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        """Returns the state of the job (mapped to readable names)."""
        info = self.get_info(job_id, mode=mode)
        return info.get("State") or "UNKNOWN"

    def read_info(self, string: tp.Union[bytes, str]) -> tp.Dict[str, tp.Dict[str, str]]:
        """Parse qstat -f output and return dict mapping job_id/_index -> stats dict.

        We normalize to keys similar to the Slurm reader: each entry contains at least
        "JobID", "State" and "NodeList".
        """
        if not isinstance(string, str):
            string = string.decode(errors="ignore")
        lines = string.splitlines()
        if not lines:
            return {}

        blocks: tp.List[tp.List[str]] = []
        current: tp.List[str] = []
        for ln in lines:
            # Start of a job block in qstat -f
            if re.match(r"^Job Id:\s*\S+", ln):
                if current:
                    blocks.append(current)
                current = [ln]
            else:
                if current is not None:
                    current.append(ln)
        if current:
            blocks.append(current)

        all_stats: tp.Dict[str, tp.Dict[str, str]] = {}
        # TODO: check this convention for PBS
        # (B for running job array ? -> https://centers.hpc.mil/users/docs/advancedTopics/Using_PBS_Job_Arrays.html)
        state_map = {
            "R": "RUNNING",
            "Q": "PENDING",
            "H": "HELD",
            "S": "SUSPENDED",
            "E": "EXITING",
            "C": "COMPLETED",
            "F": "FAILED",
            # fallback will return single-letter if unknown
        }

        for block in blocks:
            # extract job id from the first line: "Job Id: 12345.server" or "Job Id: 12345[1].server"
            first = block[0]
            m = re.match(r"^Job Id:\s*(\S+)", first)
            if not m:
                continue
            raw_jobid = m.group(1)
            # strip server suffix after dot
            raw_jobid = raw_jobid.split(".", 1)[0]
            # normalize bracketed array form "12345[1-3]" -> "12345_[1-3]" so read_job_id can parse it
            if "[" in raw_jobid and "]" in raw_jobid:
                normalized_jobid = raw_jobid.replace("[", "_[")
            else:
                normalized_jobid = raw_jobid

            # parse key = value lines
            stats: tp.Dict[str, str] = {}
            for ln in block[1:]:
                kv = re.match(r"^\s*(\S+)\s*=\s*(.*)$", ln)
                if not kv:
                    continue
                k = kv.group(1).strip()
                v = kv.group(2).strip()
                stats[k] = v

            # Prepare normalized output fields
            job_state_letter = stats.get("job_state")  # typical key
            state_val = None
            if job_state_letter:
                state_val = state_map.get(job_state_letter, job_state_letter)
            # NodeList: prefer exec_host, fall back to nodes or nodect
            node_list_raw = stats.get("exec_host") or stats.get("nodes")
            node_list_str = ""
            if node_list_raw:
                # exec_host like "node01/0+node02/0" -> "node01,node02"
                parts = re.split(r"\+|,", node_list_raw)
                nodes = []
                seen = set()
                for p in parts:
                    host = p.split("/")[0].strip()
                    if host and host not in seen:
                        seen.add(host)
                        nodes.append(host)
                node_list_str = ",".join(nodes)
            # Build the minimal stats dict returned to consumers
            out_stats: tp.Dict[str, str] = {}
            out_stats["JobID"] = raw_jobid
            if state_val:
                out_stats["State"] = state_val
            elif "job_state" in stats:
                out_stats["State"] = stats["job_state"]
            if node_list_str:
                out_stats["NodeList"] = node_list_str
            else:
                # try other fallbacks (queue host, exec_host formatted differently)
                out_stats["NodeList"] = stats.get("exec_host", "")

            # Now expand possible array job id ranges using existing read_job_id helper
            try:
                multi = read_job_id(normalized_jobid)
            except Exception as e:
                warnings.warn(f"Could not interpret {raw_jobid} correctly (please open an issue):\n{e}", DeprecationWarning)
                continue

            for split_job_id in multi:
                key = "_".join(split_job_id[:2])
                all_stats[key] = out_stats
                if len(split_job_id) >= 3:
                    # if there's a range specified, fill each index within that range
                    start = int(split_job_id[1])
                    end = int(split_job_id[2])
                    for idx in range(start, end + 1):
                        all_stats[f"{split_job_id[0]}_{idx}"] = out_stats

        return all_stats


class PBSJob(core.Job[core.R]):
    _cancel_command = "qdel"
    watcher = PBSInfoWatcher(delay_s=600)

    def _interrupt(self, timeout: bool = False) -> None:
        """Cancels the job using PBS qdel.

        Parameter
        ---------
        timeout: bool
            Ignored for PBS qdel (keeps signature compatibility).
        """
        subprocess.check_call(["qdel", self.job_id], timeout=60)


class PBSParseException(Exception):
    pass


def _expand_id_suffix(suffix_parts: str) -> tp.List[str]:
    """Parse the a suffix formatted like "1-3,5,8" into
    the list of numeric values 1,2,3,5,8.
    """
    suffixes = []
    for suffix_part in suffix_parts.split(","):
        if "-" in suffix_part:
            low, high = suffix_part.split("-")
            int_length = len(low)
            for num in range(int(low), int(high) + 1):
                suffixes.append(f"{num:0{int_length}}")
        else:
            suffixes.append(suffix_part)
    return suffixes


def _parse_node_group(node_list: str, pos: int, parsed: tp.List[str]) -> int:
    """Parse a node group of the form PREFIX[1-3,5,8] and return
    the position in the string at which the parsing stopped
    """
    prefixes = [""]
    while pos < len(node_list):
        c = node_list[pos]
        if c == ",":
            parsed.extend(prefixes)
            return pos + 1
        if c == "[":
            last_pos = node_list.index("]", pos)
            suffixes = _expand_id_suffix(node_list[pos + 1 : last_pos])
            prefixes = [prefix + suffix for prefix in prefixes for suffix in suffixes]
            pos = last_pos + 1
        else:
            for i, prefix in enumerate(prefixes):
                prefixes[i] = prefix + c
            pos += 1
    parsed.extend(prefixes)
    return pos


def _parse_node_list(node_list: str):
    try:
        pos = 0
        parsed: tp.List[str] = []
        while pos < len(node_list):
            pos = _parse_node_group(node_list, pos, parsed)
        return parsed
    except ValueError as e:
        raise PBSParseException(f"Unrecognized format for PBS_JOB_NODELIST: '{node_list}'", e) from e


class PBSJobEnvironment(job_environment.JobEnvironment):
    # Common PBS environment variables. We prefer the most widely available names,
    # but many clusters vary — the hostnames property below includes fallbacks.
    # TODO: some of these variables does not seem standard, e.g. PBS_NUM_NODES
    # Maybe we can choose a convention here and ask the user to setup hooks on his side
    # to automatically set this variables when job is launched
    # as proposed here: https://community.openpbs.org/t/get-number-of-cpus-on-allocated-job-via-environment-variable/2843/5
    # see https://2021.help.altair.com/2021.1.2/PBS%20Professional/PBSHooks2021.1.2.pdf
    _env = {
        "job_id": "PBS_JOBID",
        "num_tasks": "PBS_NP",
        "num_nodes": "PBS_NUM_NODES",
        "node": "HOSTNAME",
        "nodes": "PBS_NODEFILE",  # typically a path to a file listing nodes (one per slot)
        # MPI/OpenMPI common rank env vars as PBS doesn't always provide these directly
        "global_rank": "OMPI_COMM_WORLD_RANK",
        "local_rank": "OMPI_COMM_WORLD_LOCAL_RANK",
        # Array vars (different installations expose different names; these are common)
        "array_job_id": "PBS_ARRAY_ID",
        "array_task_id": "PBS_ARRAY_INDEX",
    }
    # # FYI: available PBS_ env variables
    # PBS_ACCOUNT=
    # PBS_ENVIRONMENT=PBS_INTERACTIVE or PBS_BATCH
    # PBS_JOBCOOKIE=
    # PBS_JOBDIR=<my-home>
    # PBS_JOBID=<job-id>
    # PBS_JOBNAME=STDIN
    # PBS_MOMPORT=15003
    # PBS_NODEFILE=/var/spool/pbs/aux/<job-id>
    # PBS_NODENUM=0
    # PBS_O_HOME=<my-home>
    # PBS_O_HOST=<submission-hostname>
    # PBS_O_LANG=en_US.UTF-8
    # PBS_O_LOGNAME=<user>
    # PBS_O_PATH=
    # PBS_O_QUEUE=<a-partition>
    # PBS_O_SHELL=/bin/bash
    # PBS_O_SYSTEM=Linux
    # PBS_O_WORKDIR=<workdir>
    # PBS_QUEUE=<another-partition>
    # PBS_TASKNUM=1
    # # following looks less PBS specific, but I don’t set them so...
    # ENVIRONMENT=BATCH
    # NCPUS=1
    # OMP_NUM_THREADS=1
    # TMPDIR=<a-tmp-dir>

    def _requeue(self, countdown: int) -> None:
        """Requeue the current job using PBS qrerun."""
        jid = self.job_id
        # qrerun is the common PBS/Torque/PBS Pro way to re-run/requeue a job.
        subprocess.check_call(["qrerun", jid], timeout=60)
        logger.get_logger().info(f"Requeued job {jid} ({countdown} remaining timeouts)")

    @property
    def hostnames(self) -> tp.List[str]:
        """Return the list of hostnames for the current PBS job.

        PBS clusters commonly expose node information in one of these ways:
        - PBS_NODEFILE: path to a file that lists each allocated slot/node (often with duplicates)
        - PBS_NODELIST / PBS_JOB_NODELIST: compressed node list, sometimes in Slurm-like bracket form
          (e.g. node[001-004]) or in '+' separated form (e.g. node001+node002)
        - A plain HOSTNAME for single-node jobs.

        This method supports those variants and normalizes the result to a list of unique hostnames
        in the order they appear.
        """
        # 1) If PBS_NODEFILE is present and points to a file, read it (one entry per slot)
        nodefile = os.environ.get(self._env["nodes"])  # PBS_NODEFILE
        if nodefile and os.path.exists(nodefile):
            try:
                with open(nodefile, "r") as fh:
                    lines = [ln.strip() for ln in fh if ln.strip()]
            except Exception:
                lines = []
            seen: tp.Set[str] = set()
            parsed: tp.List[str] = []
            for ln in lines:
                # Some nodefiles contain "node:ppn=4" or similar — take the hostname part.
                host = re.split(r"[:\s/]+", ln)[0]
                if host and host not in seen:
                    seen.add(host)
                    parsed.append(host)
            if parsed:
                return parsed

        # 2) Check for compressed node list variables commonly used
        node_vars = ("PBS_NODELIST", "PBS_JOB_NODELIST", "PBS_NODES", "NODELIST")
        for var in node_vars:
            node_list = os.environ.get(var)
            if not node_list:
                continue
            node_list = node_list.strip()
            # If it looks like the bracketed/compressed form, try to parse with _parse_node_list
            if "[" in node_list and "]" in node_list:
                try:
                    parsed = _parse_node_list(node_list)
                    if parsed:
                        return parsed
                except PBSParseException:
                    # fall back to the generic parsing below
                    pass
            # Some PBS flavors use '+' or ',' between host groups, possibly with ":ppn=" suffixes.
            parts = re.split(r"[+,]", node_list)
            parsed = []
            seen = set()
            for p in parts:
                host = p.split(":")[0].strip()
                if host and host not in seen:
                    seen.add(host)
                    parsed.append(host)
            if parsed:
                return parsed

        # 3) Fallbacks: single-host env or the hostname property from JobEnvironment
        host = os.environ.get("HOSTNAME") or os.environ.get("PBS_O_HOST")
        if host:
            return [host]
        return [self.hostname]


class PBSExecutor(core.PicklingExecutor):
    """PBS job executor
    This class is used to hold the parameters to run a job on pbs.
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
    python: Optional[str]
        Command to launch python. This allow to use singularity for example.
        Caller is responsible to provide a valid shell command here.
        By default reuse the current python executable

    Note
    ----
    - be aware that the log/output folder will be full of logs and pickled objects very fast,
      it may need cleaning.
    - the folder needs to point to a directory shared through the cluster. This is typically
      not the case for your tmp! If you try to use it, pbs will fail silently (since it
      will not even be able to log stderr.
    - use update_parameters to specify custom parameters (n_gpus etc...). If you
      input erroneous parameters, an error will print all parameters available for you.
    """

    job_class = PBSJob

    def __init__(
        self,
        folder: tp.Union[str, Path],
        max_num_timeout: int = 3,
        max_pickle_size_gb: float = 1.0,
        python: tp.Optional[str] = None
    ) -> None:
        super().__init__(
            folder,
            max_num_timeout=max_num_timeout,
            max_pickle_size_gb=max_pickle_size_gb,
        )
        self.python = shlex.quote(sys.executable) if python is None else python
        if not self.affinity() > 0:
            raise RuntimeError('Could not detect "qsub", are you indeed on a pbs cluster?')

    @classmethod
    def _equivalence_dict(cls) -> core.EquivalenceDict:
        return {
            "name": "job_name",
            "timeout_min": "time",
            "mem_gb": "mem",
            "nodes": "nodes",
            "cpus_per_task": "cpus_per_task",
            "gpus_per_node": "gpus_per_node",
            "tasks_per_node": "ntasks_per_node",
        }

    @classmethod
    def _valid_parameters(cls) -> tp.Set[str]:
        """Parameters that can be set through update_parameters"""
        return set(_get_default_parameters())

    def _convert_parameters(self, params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        params = super()._convert_parameters(params)
        # replace type in some cases
        if "mem" in params:
            params["mem"] = _convert_mem(params["mem"])
        return params

    def _internal_update_parameters(self, **kwargs: tp.Any) -> None:
        """Updates qsub submission file parameters

        Parameters
        ----------
        See pbs documentation for most parameters.
        Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
        Below are the parameters that differ from pbs documentation:

        signal_delay_s: int
            delay between the kill signal and the actual kill of the pbs job.
        setup: list
            a list of command to run in qsub before running qsub_interactive
        array_parallelism: int
            number of map tasks that will be executed in parallel

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values

        Note
        ----
        Best practice (as far as Quip is concerned): cpus_per_task=2x (number of data workers + gpus_per_task)
        You can use cpus_per_gpu=2 (requires using gpus_per_task and not gpus_per_node)
        """
        defaults = _get_default_parameters()
        in_valid_parameters = sorted(set(kwargs) - set(defaults))
        if in_valid_parameters:
            string = "\n  - ".join(f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items()))
            raise ValueError(
                f"Unavailable parameter(s): {in_valid_parameters}\nValid parameters are:\n  - {string}"
            )
        # check that new parameters are correct
        _make_qsub_string(command="nothing to do", folder=self.folder, **kwargs)
        super()._internal_update_parameters(**kwargs)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            return super()._internal_process_submissions(delayed_submissions)
        # array
        folder = utils.JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("time", 5)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        array_ex = PBSExecutor(self.folder, self.max_num_timeout)
        array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[tp.Any] = array_ex._submit_command(self._submitthem_command_str)
        tasks_ids = list(range(first_job.num_tasks))
        jobs: tp.List[core.Job[tp.Any]] = [
            PBSJob(folder=self.folder, job_id=f"{first_job.job_id}_{a}", tasks=tasks_ids) for a in range(n)
        ]
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs

    @property
    def _submitthem_command_str(self) -> str:
        return " ".join([self.python, "-u -m submitthem.core._submit", shlex.quote(str(self.folder))])

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        return _make_qsub_string(command=command, folder=self.folder, **self.parameters)

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self, submission_file_path: Path) -> tp.List[str]:
        return ["qsub", str(submission_file_path)]

    @staticmethod
    def _get_job_id_from_submission_command(string: tp.Union[bytes, str]) -> str:
        """Returns the job ID from the output of qsub string"""
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"job (?P<id>[0-9]+)", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of qsub output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")

    @classmethod
    def affinity(cls) -> int:
        return -1 if shutil.which("qsub") is None else 2


@functools.lru_cache()
def _get_default_parameters() -> tp.Dict[str, tp.Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_qsub_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command", "folder", "map_count"}}


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_qsub_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitthem",
    partition: tp.Optional[str] = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: tp.Optional[int] = None,
    cpus_per_task: tp.Optional[int] = None,
    cpus_per_gpu: tp.Optional[int] = None,
    num_gpus: tp.Optional[int] = None,  # legacy
    gpus_per_node: tp.Optional[int] = None,
    gpus_per_task: tp.Optional[int] = None,
    qos: tp.Optional[str] = None,  # quality of service
    setup: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    mem_per_gpu: tp.Optional[str] = None,
    mem_per_cpu: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: tp.Optional[str] = None,
    constraint: tp.Optional[str] = None,
    exclude: tp.Optional[str] = None,
    account: tp.Optional[str] = None,
    gres: tp.Optional[str] = None,
    mail_type: tp.Optional[str] = None,
    mail_user: tp.Optional[str] = None,
    nodelist: tp.Optional[str] = None,
    dependency: tp.Optional[str] = None,
    exclusive: tp.Optional[tp.Union[bool, str]] = None,
    array_parallelism: int = 256,
    wckey: str = "submitthem",
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    qsub_interactive_args: tp.Optional[tp.Iterable[str]] = None,
    use_qsub_interactive: bool = True,
) -> str:
    """Creates the content of a qsub file with provided parameters

    Parameters
    ----------
    See pbs qsub documentation for most parameters:
    https://help.altair.com/2022.1.0/PBS%20Professional/PBSReferenceGuide2022.1.pdf

    Below are the parameters that differ from pbs documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the kill signal and the actual kill of the pbs job.
    setup: list
        a list of command to run in qsub before running qsub_interactive
    map_size: int
        number of simultaneous map/array jobs allowed
    additional_parameters: dict
        Forces any parameter to a given value in qsub. This can be useful
        to add parameters which are not currently available in submitthem.
        Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    qsub_interactive_args: List[str]
        Add each argument in the list to the `qsub -I` call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonpbs = [
        "nonpbs",
        "folder",
        "command",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "setup",
        "signal_delay_s",
        "stderr_to_stdout",
        "qsub_interactive_args",
        "use_qsub_interactive",  # if False, use python directly in qsub instead of through `qsub -I`
    ]
    parameters = {k: v for k, v in locals().items() if v is not None and k not in nonpbs}
    ### rename and reformat parameters

    #
    # remove --signal option as there is no equivalent for PBS
    # TODO: build an alternative with qalter or timeout command
    # parameters["signal"] = f"{PBSJobEnvironment.USR_SIG}@{signal_delay_s}"

    # replace any slurm option by select
    if "nodes" in parameters:
        num_nodes = parameters.pop("nodes")
        # Build select clause: nodes with optional task and cpu specifications
        # PBS format: select=<nodes>:ncpus=<cpus_per_node>:mpiprocs=<tasks_per_node>
        select_parts = [str(num_nodes)]

        # Add CPU specification
        ncpus_val = parameters.pop("cpus_per_task", None)
        if ncpus_val is not None:
            select_parts.append(f"ncpus={ncpus_val}")
        else:
            select_parts.append("ncpus=1")

        # Add MPI process specification (ntasks per node)
        if ntasks_per_node is not None:
            select_parts.append(f"mpiprocs={ntasks_per_node}")
            if "ntasks_per_node" in parameters:
                parameters.pop("ntasks_per_node")

        parameters["l select"] = ":".join(select_parts)
    elif "ntasks_per_node" in parameters:
        # If ntasks_per_node is set but not nodes, add mpiprocs to existing select
        mpiprocs = parameters.pop("ntasks_per_node")
        if "l select" in parameters:
            parameters["l select"] += f":mpiprocs={mpiprocs}"
        else:
            parameters["l select"] = f"1:mpiprocs={mpiprocs}"

    if "gpus_per_node" in parameters:
        gpus = parameters.pop("gpus_per_node")
        parameters["l select"] += f":ngpus={gpus}"

    if "mem" in parameters:
        mem = parameters.pop('mem')
        # Parse memory value - handle both numeric and string formats (e.g., "4", "4GB", "512MB")
        mem_val: float = 0.0
        if isinstance(mem, str):
            # Extract numeric part
            match = re.match(r'(\d+(?:\.\d+)?)', mem)
            mem_val = float(match.group(1)) if match else 0.0
        else:
            mem_val = float(mem) if mem is not None else 0.0

        # Format memory for PBS
        if mem_val > 0:
            parameters["l select"] += f":mem={int(mem_val)}gb"

    # Handle memory per GPU (convert to PBS format)
    if "mem_per_gpu" in parameters:
        mem_per_gpu = parameters.pop("mem_per_gpu")
        if "l select" in parameters:
            parameters["l select"] += f":mem_per_gpu={mem_per_gpu}"
        else:
            parameters["l select"] = f"1:mem_per_gpu={mem_per_gpu}"

    # Handle memory per CPU (convert to PBS format)
    if "mem_per_cpu" in parameters:
        mem_per_cpu = parameters.pop("mem_per_cpu")
        if "l select" in parameters:
            parameters["l select"] += f":mem_per_cpu={mem_per_cpu}"
        else:
            parameters["l select"] = f"1:mem_per_cpu={mem_per_cpu}"

    if exclusive:
        parameters.pop("exclusive")
        parameters["l place"] = "excl"

    if partition is not None:
        parameters.pop("partition")
        parameters["q"] = partition

    # Handle gpus_per_task (convert to PBS format)
    if "gpus_per_task" in parameters:
        gpus_per_task = parameters.pop("gpus_per_task")
        # In PBS, gpus_per_task is typically represented as ngpus in select with proper constraints
        if "l select" in parameters:
            parameters["l select"] += f":ngpus={gpus_per_task}"
        else:
            parameters["l select"] = f"1:ngpus={gpus_per_task}"

    #
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn('"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")')

    # Remove legacy num_gpus parameter and warn if used
    if "num_gpus" in parameters:
        _ = parameters.pop("num_gpus")  # Extract and discard legacy parameter
        warnings.warn(
            '"num_gpus" is deprecated. Use "gpus_per_node", "gpus_per_task", '
            'or "gres" instead for PBS compatibility.',
            DeprecationWarning,
            stacklevel=2
        )

    # add necessary parameters
    paths = utils.JobPaths(folder=folder)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)
    # Job arrays will write files in the form  <ARRAY_ID>_<ARRAY_TASK_ID>_<TASK_ID>
    if map_count is not None:
        assert isinstance(map_count, int) and map_count
        parameters["J"] = f"0-{map_count - 1}%{min(map_count, array_parallelism)}"
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["o"] = stdout.replace("%t", "0")
    if not stderr_to_stdout:
        parameters["e"] = stderr.replace("%t", "0")
        parameters["j"] = "oe"

    #
    # remove --open-mode option as there is no equivalent for PBS
    # parameters["open-mode"] = "append"

    #
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(_as_qsub_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup
    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass -o and -e here, (TODO: check this statement for PBS: because the qsub command doesn't work as expected with a filename pattern)

    if use_qsub_interactive:
        # using `qsub -I` has been the only option historically,
        # TODO: check next statement for PBS
        # but it's not clear anymore if it is necessary, and using it prevents
        # jobs from scheduling other jobs
        stderr_flags = [] if stderr_to_stdout else ["-e", stderr]
        if qsub_interactive_args is None:
            qsub_interactive_args = []
        # TODO: modify for PBS -> remove --unbuffered option as there is no equivalent for PBS
        qsub_interactive_cmd = _shlex_join(["qsub", "-I", "-o", stdout, *stderr_flags, *qsub_interactive_args])
        command = " ".join((qsub_interactive_cmd, command))

    lines += [
        "",
        "# command",
        "export SUBMITTHEM_EXECUTOR=pbs",
        # The input "command" is supposed to be a valid shell command
        command,
        "",
    ]
    return "\n".join(lines)


def _convert_mem(mem_gb: float) -> str:
    if mem_gb == int(mem_gb):
        return f"{int(mem_gb)}GB"
    return f"{int(mem_gb * 1024)}MB"


def _as_qsub_flag(key: str, value: tp.Any) -> str:
    key = key.replace("_", "-")
    if value is True:
        return f"#PBS -{key}"

    value = shlex.quote(str(value))
    if key.startswith('l '):
        return f"#PBS -{key}={value}"
    else:
        return f"#PBS -{key} {value}"


def _shlex_join(split_command: tp.List[str]) -> str:
    """Same as shlex.join, but that was only added in Python 3.8"""
    return " ".join(shlex.quote(arg) for arg in split_command)
