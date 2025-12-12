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
    # Updated pattern to better handle comma-separated ranges like "2,4-12"
    pattern = r"(?P<main_id>\d+)_\[(?P<arrays>[0-9,\-]+)(\%\d+)?\]"
    match = re.search(pattern, job_id)
    if match is not None:
        main = match.group("main_id")
        arrays_str = match.group("arrays")
        result = []
        for array_range in arrays_str.split(","):
            array_range = array_range.strip()
            if "-" in array_range:
                parts = array_range.split("-")
                result.append(tuple([main] + parts))
            else:
                result.append((main, array_range))
        return result
    else:
        main_id, *array_id = job_id.split("_", 1)
        if not array_id:
            return [(main_id,)]
        # there is an array
        # Strip throttle notation (e.g., "4%3" -> "4")
        array_str = array_id[0].split("%")[0]
        array_num = str(int(array_str))  # Validate it's an integer
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
        """Parse qstat output and return dict mapping job_id/_index -> stats dict.

        Handles multiple formats:
        1. qstat -f format (full, multi-line): Job Id: X, job_state = Y
        2. qstat format (simple, column-based): Fixed-width columns with Job ID, Username, Queue, etc.

        We normalize to keys similar to the Slurm reader: each entry contains at least
        "JobID", "State" and "NodeList".
        """
        if not isinstance(string, str):
            string = string.decode(errors="ignore")
        lines = string.splitlines()
        if not lines:
            return {}

        state_map = {
            "R": "RUNNING",
            "Q": "PENDING",
            "H": "HELD",
            "S": "SUSPENDED",
            "E": "EXITING",
            "C": "COMPLETED",
            "F": "FAILED",
            "X": "EXITING",
            "U": "UNKNOWN",
            # fallback will return single-letter if unknown
        }

        # Check if this is qstat -f format (starts with "Job Id:" - lowercase 'd')
        # Skip leading empty lines to find the first non-empty line
        for line in lines:
            if line.strip():
                if re.match(r"^Job Id:\s*\S+", line.strip()):
                    return self._read_info_qstat_f_format(lines, state_map)
                break

        # Otherwise, parse qstat format (simple column-based format with "Job ID" - uppercase 'D')
        return self._read_info_qstat_format(lines, state_map)

    def _read_info_qstat_f_format(self, lines: tp.List[str], state_map: tp.Dict[str, str]) -> tp.Dict[str, tp.Dict[str, str]]:
        """Parse qstat -f format output (full format, multi-line blocks)"""
        all_stats: tp.Dict[str, tp.Dict[str, str]] = {}

        # Split input into job blocks
        blocks: tp.List[tp.List[str]] = []
        current: tp.List[str] = []
        for ln in lines:
            # Start of a job block in qstat -f
            if re.match(r"^Job Id:\s*\S+", ln):
                if current:
                    blocks.append(current)
                current = [ln]
            else:
                if current:
                    current.append(ln)
        if current:
            blocks.append(current)

        for block in blocks:
            if not block:
                continue

            # Extract job id from the first line: "Job Id: 12345.server" or "Job Id: 12345[1].server"
            first = block[0]
            m = re.match(r"^Job Id:\s*(\S+)", first)
            if not m:
                continue
            raw_jobid = m.group(1)
            # Strip server suffix after dot
            raw_jobid = raw_jobid.split(".", 1)[0]
            # Normalize bracketed array form "12345[1-3]" -> "12345_[1-3]"
            # Only add underscore if not already present
            normalized_jobid = raw_jobid.replace("[", "_[") if "[" in raw_jobid and "_[" not in raw_jobid else raw_jobid
            # Normalize JobID in output: add underscore before brackets if not already present
            output_jobid = raw_jobid.replace("[", "_[") if "[" in raw_jobid and "_[" not in raw_jobid else raw_jobid

            # Parse key = value lines
            stats: tp.Dict[str, str] = {}
            for ln in block[1:]:
                kv = re.match(r"^\s*(\S+)\s*=\s*(.*)$", ln)
                if not kv:
                    continue
                k = kv.group(1).strip()
                v = kv.group(2).strip()
                stats[k] = v

            # Get job state
            job_state_letter = stats.get("job_state", "")
            state_val = state_map.get(job_state_letter, job_state_letter) if job_state_letter else "UNKNOWN"

            # Get node list
            node_list_raw = stats.get("exec_host") or stats.get("nodes")
            node_list_str = ""
            if node_list_raw:
                # exec_host like "node01/0+node02/0" -> "node01,node02"
                parts = re.split(r"\+|,", node_list_raw)
                nodes = []
                seen: tp.Set[str] = set()
                for p in parts:
                    host = p.split("/")[0].strip()
                    if host and host not in seen:
                        seen.add(host)
                        nodes.append(host)
                node_list_str = ",".join(nodes)

            # Parse the job ID to get main job and array indices
            try:
                multi = read_job_id(normalized_jobid)
            except Exception as e:
                warnings.warn(f"Could not interpret {raw_jobid} correctly (please open an issue):\n{e}", DeprecationWarning)
                continue

            # Expand array ranges
            for split_job_id in multi:
                main_id = split_job_id[0]
                if len(split_job_id) == 1:
                    # Non-array job
                    key = main_id
                    out_stats: tp.Dict[str, str] = {"JobID": output_jobid, "State": state_val}
                    if node_list_str:
                        out_stats["NodeList"] = node_list_str
                    all_stats[key] = out_stats
                elif len(split_job_id) == 2:
                    # Single array task
                    array_idx = split_job_id[1]
                    key = f"{main_id}_{array_idx}"
                    out_stats = {"JobID": output_jobid, "State": state_val}
                    if node_list_str:
                        out_stats["NodeList"] = node_list_str
                    all_stats[key] = out_stats
                elif len(split_job_id) >= 3:
                    # Array range - expand it
                    start = int(split_job_id[1])
                    end = int(split_job_id[2])
                    for idx in range(start, end + 1):
                        key = f"{main_id}_{idx}"
                        out_stats = {"JobID": output_jobid, "State": state_val}
                        if node_list_str:
                            out_stats["NodeList"] = node_list_str
                        all_stats[key] = out_stats

        return all_stats

    def _read_info_qstat_format(self, lines: tp.List[str], state_map: tp.Dict[str, str]) -> tp.Dict[str, tp.Dict[str, str]]:
        """Parse simple qstat format with fixed-width columns.

        Format:
                                                                    Req'd  Req'd   Elap
        Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
        --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----

        Or minimal format:
        JobID            S
        5610980          R
        """
        all_stats: tp.Dict[str, tp.Dict[str, str]] = {}

        if not lines:
            return all_stats

        # Find the separator line (contains dashes) if it exists
        separator_idx = -1
        for i, line in enumerate(lines):
            if re.match(r"^\s*-+\s*-+", line):
                separator_idx = i
                break

        # Determine data start index
        if separator_idx > 0:
            # Standard format with separator: header at separator_idx - 1, data after separator
            header_line = lines[separator_idx - 1]
            data_start_idx = separator_idx + 1
        elif lines:
            # Simple format without separator: first line is header, data starts at line 1
            header_line = lines[0]
            data_start_idx = 1
        else:
            return all_stats

        if not header_line:
            return all_stats

        # Find column positions by looking for "Job ID" (the standard qstat format)
        jobid_pos = header_line.find("Job ID")
        # Also accept "JobID" as fallback for test/simplified formats
        if jobid_pos < 0:
            jobid_pos = header_line.find("JobID")
        state_pos = header_line.rfind(" S ")  # State column is marked with 'S'

        if jobid_pos < 0:
            return all_stats

        # Parse data rows
        for line in lines[data_start_idx:]:
            if not line.strip():
                continue

            # Extract job ID from fixed position
            if len(line) > jobid_pos:
                # Find the end of the job ID (next whitespace or fixed width)
                jobid_end = jobid_pos
                while jobid_end < len(line) and not line[jobid_end].isspace():
                    jobid_end += 1
                raw_jobid = line[jobid_pos:jobid_end].strip()
                if not raw_jobid:
                    continue

                # Skip entries with dots or plus signs (PBS metadata entries like 5610980.ext+ or 5610980.0)
                if "." in raw_jobid or "+" in raw_jobid:
                    continue
            else:
                continue

            # Extract state - look for the single character in the S column
            state_letter = ""
            if state_pos >= 0 and len(line) > state_pos:
                # State should be a single non-space character around the S column
                state_letter = line[state_pos:state_pos + 2].strip()
                if len(state_letter) > 1:
                    state_letter = state_letter[0]

            # If we didn't find state, try to extract from the line by looking for single-letter codes
            if not state_letter or state_letter not in state_map:
                # Try alternative: look for known state letters in the latter part of the line
                for part in line.split():
                    if len(part) == 1 and part in state_map:
                        state_letter = part
                        break

            if not state_letter or state_letter not in state_map:
                state_letter = "U"  # Default to UNKNOWN if we can't determine state

            state_val = state_map.get(state_letter, state_letter)

            # Normalize bracketed array form "12345[1-3]" -> "12345_[1-3]"
            # Only add underscore if not already present
            normalized_jobid = raw_jobid.replace("[", "_[") if "[" in raw_jobid and "_[" not in raw_jobid else raw_jobid

            # Parse the job ID to get main job and array indices
            try:
                multi = read_job_id(normalized_jobid)
            except Exception as e:
                warnings.warn(f"Could not interpret {raw_jobid} correctly (please open an issue):\n{e}", DeprecationWarning)
                continue

            # Normalize JobID in output: add underscore before brackets if not already present
            output_jobid = raw_jobid.replace("[", "_[") if "[" in raw_jobid and "_[" not in raw_jobid else raw_jobid

            # Expand array ranges
            for split_job_id in multi:
                main_id = split_job_id[0]
                if len(split_job_id) == 1:
                    # Non-array job
                    key = main_id
                    out_stats = {"JobID": output_jobid, "State": state_val}
                    all_stats[key] = out_stats
                elif len(split_job_id) == 2:
                    # Single array task
                    array_idx = split_job_id[1]
                    key = f"{main_id}_{array_idx}"
                    out_stats = {"JobID": output_jobid, "State": state_val}
                    all_stats[key] = out_stats
                elif len(split_job_id) >= 3:
                    # Array range - expand it
                    start = int(split_job_id[1])
                    end = int(split_job_id[2])
                    for idx in range(start, end + 1):
                        key = f"{main_id}_{idx}"
                        out_stats = {"JobID": output_jobid, "State": state_val}
                        all_stats[key] = out_stats

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
        suffix_part = suffix_part.strip()
        if not suffix_part:
            raise PBSParseException(f"Empty suffix in '{suffix_parts}'")
        if "-" in suffix_part:
            parts = suffix_part.split("-")
            if len(parts) != 2:
                raise PBSParseException(f"Invalid range format in '{suffix_parts}': '{suffix_part}'")
            low, high = parts
            if not low or not high:
                raise PBSParseException(f"Invalid range format in '{suffix_parts}': '{suffix_part}' (missing start or end)")
            try:
                low_int = int(low)
                high_int = int(high)
            except ValueError as e:
                raise PBSParseException(f"Non-numeric values in range '{suffix_part}': {e}") from e
            int_length = len(low)
            for num in range(low_int, high_int + 1):
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
                # Bracketed format must be valid - don't fall back on error
                parsed = _parse_node_list(node_list)
                if parsed:
                    return parsed
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

    # Merge additional_parameters early so they go through conversion logic
    if additional_parameters is not None:
        parameters.update(additional_parameters)

    # Build the select clause incrementally with all resource specifications
    select_clause = ""

    # Start with nodes specification
    if "nodes" in parameters:
        num_nodes = parameters.pop("nodes")
        select_clause = str(num_nodes)
    else:
        select_clause = "1"  # Default to 1 node if not specified

    # Add CPU specification (cpus_per_task or cpus_per_node)
    ncpus_val = parameters.pop("cpus_per_task", None)
    if ncpus_val is not None:
        select_clause += f":ncpus={ncpus_val}"
    else:
        select_clause += ":ncpus=1"

    # Add MPI process specification (ntasks_per_node / mpiprocs)
    ntasks_val = parameters.pop("ntasks_per_node", None)
    if ntasks_val is not None:
        select_clause += f":mpiprocs={ntasks_val}"

    # Add GPU specifications
    gpus_per_node_val = parameters.pop("gpus_per_node", None)
    if gpus_per_node_val is not None:
        select_clause += f":ngpus={gpus_per_node_val}"

    gpus_per_task_val = parameters.pop("gpus_per_task", None)
    if gpus_per_task_val is not None:
        select_clause += f":ngpus={gpus_per_task_val}"

    # Add memory specifications
    mem_val = parameters.pop("mem", None)
    if mem_val is not None:
        # Parse memory value - handle both numeric and string formats (e.g., "4", "4GB", "512MB")
        mem_num: float = 0.0
        if isinstance(mem_val, str):
            # Extract numeric part
            match = re.match(r'(\d+(?:\.\d+)?)', mem_val)
            mem_num = float(match.group(1)) if match else 0.0
        else:
            mem_num = float(mem_val) if mem_val is not None else 0.0

        if mem_num > 0:
            select_clause += f":mem={int(mem_num)}gb"

    mem_per_gpu_val = parameters.pop("mem_per_gpu", None)
    if mem_per_gpu_val is not None:
        select_clause += f":mem_per_gpu={mem_per_gpu_val}"

    mem_per_cpu_val = parameters.pop("mem_per_cpu", None)
    if mem_per_cpu_val is not None:
        select_clause += f":mem_per_cpu={mem_per_cpu_val}"

    # Set the select clause
    parameters["l select"] = select_clause

    # Handle job name (rename job_name -> N)
    if "job_name" in parameters:
        job_name_val = parameters.pop("job_name")
        parameters["N"] = job_name_val

    # Handle time/walltime (convert time minutes to walltime)
    if "time" in parameters:
        time_min = parameters.pop("time")
        # Convert minutes to walltime format HH:MM:SS
        hours = time_min // 60
        minutes = time_min % 60
        parameters["l walltime"] = f"{hours:02d}:{minutes:02d}:00"

    # Handle QoS (convert qos -> qos in resource list)
    if "qos" in parameters:
        qos_val = parameters.pop("qos")
        parameters["l qos"] = qos_val

    # Handle placement constraints (exclusive)
    if exclusive:
        parameters.pop("exclusive", None)
        parameters["l place"] = "excl"

    # Handle partition/queue
    if partition is not None:
        parameters.pop("partition", None)
        parameters["q"] = partition

    # Handle cpus_per_gpu warning
    if "cpus_per_gpu" in parameters:
        parameters.pop("cpus_per_gpu")
        if gpus_per_task_val is None:
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
    """Convert parameter key-value pair to PBS qsub directive format.

    PBS uses format: #PBS -flag value or #PBS -flag=value
    Different flags have different conventions.
    """
    # Handle special key mappings for PBS compatibility
    pbs_flag_map = {
        'job-name': 'N',
        'output': 'o',
        'error': 'e',
        'join': 'j',
        'J': 'J',  # Array job range
        'select': 'l select',
        'place': 'l place',
        'walltime': 'l walltime',
    }

    # Map the key if it's in our special map
    if key in pbs_flag_map:
        key = pbs_flag_map[key]
    else:
        # Replace underscores with hyphens for other keys
        key = key.replace("_", "-")

    # Handle boolean flags
    if value is True:
        return f"#PBS -{key}"

    value_str = shlex.quote(str(value))

    # Different formatting for different flag types
    if key.startswith('l ') or key.startswith('W '):
        # Resource list and -W flags use -flag key=value format
        return f"#PBS -{key}={value_str}"
    else:
        # Other flags use -flag value format
        return f"#PBS -{key} {value_str}"


def _shlex_join(split_command: tp.List[str]) -> str:
    """Same as shlex.join, but that was only added in Python 3.8"""
    return " ".join(shlex.quote(arg) for arg in split_command)
