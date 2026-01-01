# pylint: disable=duplicate-code
# LSF executor intentionally mirrors Slurm patterns for consistency.

import functools
import os
import re
import shlex
import shutil
import subprocess
import sys
import typing as tp
import uuid
from pathlib import Path

from ..core import core, job_environment, logger, utils


def read_job_id(job_id: str) -> tp.List[tp.Tuple[str, ...]]:
    """Reads formatted job id and returns a tuple with format:
    (main_id, [array_index, [final_array_index])

    LSF array jobs use format like: 12345[1], 12345[1-10], etc.
    Submitit internally uses underscore format: 12345_1
    """
    # Handle submitit internal format: 12345_1
    if "_" in job_id:
        main_id, array_id = job_id.split("_", 1)
        return [(main_id, array_id)]
    # Handle LSF bracket format: 12345[1-10]
    pattern = r"(?P<main_id>\d+)\[(?P<arrays>(\d+(-\d+)?(,)?)+)\]"
    match = re.search(pattern, job_id)
    if match is not None:
        main = match.group("main_id")
        array_ranges = match.group("arrays").split(",")
        return [tuple([main] + array_range.split("-")) for array_range in array_ranges]
    # Simple job id
    return [(job_id,)]


class LsfInfoWatcher(core.InfoWatcher):
    """Watches LSF job status using bjobs command."""

    def _make_command(self) -> tp.Optional[tp.List[str]]:
        # Get unique parent job IDs (for arrays, just the main ID)
        to_check = {x.split("_")[0] for x in self._registered - self._finished}
        if not to_check:
            return None
        # Use bjobs with specific output format for easier parsing
        # -o specifies output format, -noheader removes header line
        # JOBINDEX is needed to distinguish array job elements
        command = ["bjobs", "-o", "JOBID JOBINDEX STAT", "-noheader"]
        for jid in to_check:
            command.append(str(jid))
        return command

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        """Returns the state of the job.
        State of finished jobs are cached (use watcher.clear() to remove all cache)

        Parameters
        ----------
        job_id: str
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        info = self.get_info(job_id, mode=mode)
        return info.get("State") or "UNKNOWN"

    def read_info(self, string: tp.Union[bytes, str]) -> tp.Dict[str, tp.Dict[str, str]]:
        """Reads the output of bjobs and returns a dictionary containing main information.

        Preferred format from: bjobs -o "JOBID JOBINDEX STAT" -noheader
        - Non-array job: "301828 0 DONE" (JOBINDEX=0)
        - Array job element: "301970 1 RUN" (1-based index)
        """
        if not isinstance(string, str):
            string = string.decode()
        lines = string.strip().splitlines()
        if not lines:
            return {}

        all_stats: tp.Dict[str, tp.Dict[str, str]] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # Preferred 3-column format: JOBID JOBINDEX STAT
            if len(parts) >= 3:
                self._parse_three_column_format(parts, all_stats)
                continue

            # Fallback 2-column format: JOBID STAT (legacy or bracket format)
            if len(parts) == 2:
                self._parse_two_column_format(parts, all_stats)

        return all_stats

    def _parse_three_column_format(
        self, parts: tp.List[str], all_stats: tp.Dict[str, tp.Dict[str, str]]
    ) -> None:
        """Parse 3-column bjobs output: JOBID JOBINDEX STAT."""
        job_id_raw = parts[0]
        job_index = parts[1]
        state_raw = parts[2]
        state = self._normalize_state(state_raw)

        # JOBINDEX=0 means non-array job, JOBINDEX>0 means array element (1-based)
        if job_index == "0":
            all_stats[job_id_raw] = {"JobID": job_id_raw, "State": state}
        else:
            submitit_job_id = f"{job_id_raw}_{job_index}"
            all_stats[submitit_job_id] = {"JobID": f"{job_id_raw}[{job_index}]", "State": state}
            # Also store under the main ID for queries that don't specify index
            if job_id_raw not in all_stats:
                all_stats[job_id_raw] = {"JobID": job_id_raw, "State": state}

    def _parse_two_column_format(
        self, parts: tp.List[str], all_stats: tp.Dict[str, tp.Dict[str, str]]
    ) -> None:
        """Parse 2-column bjobs output: JOBID STAT (legacy or bracket format)."""
        job_id_raw = parts[0]
        state_raw = parts[1]
        state = self._normalize_state(state_raw)

        if "[" in job_id_raw:
            match = re.match(r"(\d+)\[(\d+)\]", job_id_raw)
            if match:
                main_id = match.group(1)
                array_idx = match.group(2)
                submitit_job_id = f"{main_id}_{array_idx}"
                all_stats[submitit_job_id] = {"JobID": job_id_raw, "State": state}
                if main_id not in all_stats:
                    all_stats[main_id] = {"JobID": job_id_raw, "State": state}
        else:
            all_stats[job_id_raw] = {"JobID": job_id_raw, "State": state}

    @staticmethod
    def _normalize_state(lsf_state: str) -> str:
        """Normalize LSF job states to submitit-compatible states."""
        # LSF states: PEND, RUN, DONE, EXIT, PSUSP, USUSP, SSUSP, WAIT, ZOMBI
        state_map = {
            "PEND": "PENDING",
            "RUN": "RUNNING",
            "DONE": "COMPLETED",
            "EXIT": "FAILED",
            "PSUSP": "SUSPENDED",
            "USUSP": "SUSPENDED",
            "SSUSP": "SUSPENDED",
            "WAIT": "PENDING",
            "ZOMBI": "FAILED",
            "UNKWN": "UNKNOWN",
        }
        return state_map.get(lsf_state.upper(), "UNKNOWN")


class LsfJob(core.Job[core.R]):
    """LSF job handle."""

    _cancel_command = "bkill"
    watcher = LsfInfoWatcher(delay_s=60)

    def _interrupt(self, timeout: bool = False) -> None:
        """Sends preemption or timeout signal to the job (for testing purpose)

        Parameter
        ---------
        timeout: bool
            Whether to trigger a job time-out (if False, it triggers preemption)
        """
        # In case of preemption, SIGTERM is sent first (same as Slurm behavior)
        if not timeout:
            subprocess.check_call(["bkill", "-s", "SIGTERM", self.job_id])
        subprocess.check_call(["bkill", "-s", LsfJobEnvironment.USR_SIG, self.job_id])


class LsfParseException(Exception):
    """Exception raised when parsing LSF output fails."""


class LsfJobEnvironment(job_environment.JobEnvironment):
    """LSF job environment for running jobs."""

    _env = {
        "job_id": "LSB_JOBID",
        "num_tasks": "SUBMITIT_LSF_NTASKS",
        "num_nodes": "SUBMITIT_LSF_NNODES",
        "node": "SUBMITIT_LSF_NODEID",
        "global_rank": "SUBMITIT_LSF_GLOBAL_RANK",
        "local_rank": "SUBMITIT_LSF_LOCAL_RANK",
        "array_job_id": "SUBMITIT_LSF_ARRAY_JOB_ID",
        "array_task_id": "SUBMITIT_LSF_ARRAY_TASK_ID",
    }

    def _requeue(self, countdown: int) -> None:
        """Requeue the current job using brequeue."""
        jid = self.job_id
        # For array jobs, we need to specify the element: brequeue 12345[7]
        if self.array_job_id and self.array_task_id:
            lsf_job_ref = f"{self.array_job_id}[{self.array_task_id}]"
        else:
            lsf_job_ref = self.raw_job_id
        subprocess.check_call(["brequeue", lsf_job_ref], timeout=60)
        logger.get_logger().info(f"Requeued job {jid} ({countdown} remaining timeouts)")

    @property
    def hostnames(self) -> tp.List[str]:
        """Get the list of hostnames for the job."""
        # LSB_HOSTS contains space-separated list of hosts
        hosts = os.environ.get("LSB_HOSTS", "")
        if not hosts:
            return [self.hostname]
        return hosts.split()


class LsfExecutor(core.PicklingExecutor):
    """LSF job executor.

    This class is used to hold the parameters to run a job on LSF.
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
        Command to launch python. This allows using singularity for example.
        Caller is responsible to provide a valid shell command here.
        By default reuses the current python executable.

    Note
    ----
    - be aware that the log/output folder will be full of logs and pickled objects very fast,
      it may need cleaning.
    - the folder needs to point to a directory shared through the cluster. This is typically
      not the case for your tmp! If you try to use it, LSF will fail silently (since it
      will not even be able to log stderr).
    - use update_parameters to specify custom parameters (gpus_per_node etc...). If you
      input erroneous parameters, an error will print all parameters available for you.
    """

    job_class = LsfJob

    def __init__(
        self,
        folder: tp.Union[str, Path],
        max_num_timeout: int = 3,
        max_pickle_size_gb: float = 1.0,
        python: tp.Optional[str] = None,
    ) -> None:
        super().__init__(
            folder,
            max_num_timeout=max_num_timeout,
            max_pickle_size_gb=max_pickle_size_gb,
        )
        self.python = shlex.quote(sys.executable) if python is None else python
        if not self.affinity() > 0:
            raise RuntimeError('Could not detect "bsub", are you indeed on an LSF cluster?')

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
        # Convert mem_gb to LSF format (e.g., "4GB" or "4096MB")
        if "mem" in params:
            params["mem"] = _convert_mem(params["mem"])
        return params

    def _internal_update_parameters(self, **kwargs: tp.Any) -> None:
        """Updates bsub submission file parameters.

        Parameters
        ----------
        See LSF bsub documentation for most parameters.
        Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, queue

        Below are the parameters that differ from LSF documentation:

        signal_delay_s: int
            delay between the warning signal and the actual kill of the LSF job.
        setup: list
            a list of commands to run in bsub before running the main command
        array_parallelism: int
            number of map tasks that will be executed in parallel

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values
        """
        defaults = _get_default_parameters()
        invalid_parameters = sorted(set(kwargs) - set(defaults))
        if invalid_parameters:
            string = "\n  - ".join(f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items()))
            raise ValueError(
                f"Unavailable parameter(s): {invalid_parameters}\nValid parameters are:\n  - {string}"
            )
        # Check that new parameters are correct
        _make_bsub_string(command="nothing to do", folder=self.folder, **kwargs)
        super()._internal_update_parameters(**kwargs)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            return super()._internal_process_submissions(delayed_submissions)
        # Array job
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
        array_ex = LsfExecutor(self.folder, self.max_num_timeout)
        array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[tp.Any] = array_ex._submit_command(self._submitit_command_str)
        tasks_ids = list(range(first_job.num_tasks))
        jobs: tp.List[core.Job[tp.Any]] = [
            # LSF arrays are 1-based, so indices go from 1 to n
            LsfJob(folder=self.folder, job_id=f"{first_job.job_id}_{a}", tasks=tasks_ids)
            for a in range(1, n + 1)
        ]
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs

    @property
    def _submitit_command_str(self) -> str:
        return " ".join([self.python, "-u -m submitit.core._submit", shlex.quote(str(self.folder))])

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        return _make_bsub_string(command=command, folder=self.folder, **self.parameters)

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self, submission_file_path: Path) -> tp.List[str]:
        return ["bsub", "<", str(submission_file_path)]

    def _submit_command(self, command: str) -> core.Job[tp.Any]:
        """Submits a command to the cluster.

        Override to handle LSF's bsub stdin-based submission.
        """
        tmp_uuid = uuid.uuid4().hex
        submission_file_path = (
            utils.JobPaths.get_first_id_independent_folder(self.folder) / f".submission_file_{tmp_uuid}.sh"
        )
        submission_file_path.parent.mkdir(parents=True, exist_ok=True)
        with submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command, tmp_uuid))

        # LSF uses stdin for bsub, so we need to handle it differently
        with submission_file_path.open("r") as f:
            try:
                output = subprocess.check_output(["bsub"], stdin=f, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise utils.FailedSubmissionError(
                    f"bsub submission failed with return code {e.returncode}:\n{e.output.decode()}"
                ) from e

        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))
        job: core.Job[tp.Any] = self.job_class(folder=self.folder, job_id=job_id, tasks=tasks_ids)
        job.paths.move_temporary_file(submission_file_path, "submission_file", keep_as_symlink=True)
        self._write_job_id(job.job_id, tmp_uuid)
        self._set_job_permissions(job.paths.folder)
        return job

    @staticmethod
    def _get_job_id_from_submission_command(string: tp.Union[bytes, str]) -> str:
        """Returns the job ID from the output of bsub string.

        LSF typically outputs: Job <12345> is submitted to queue <normal>.
        """
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"Job <(?P<id>\d+)>", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of bsub output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")

    @classmethod
    def affinity(cls) -> int:
        return -1 if shutil.which("bsub") is None else 2


@functools.lru_cache()
def _get_default_parameters() -> tp.Dict[str, tp.Any]:
    """Parameters that can be set through update_parameters"""
    specs = __import__("inspect").getfullargspec(_make_bsub_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command", "folder", "map_count"}}


def _bsub_job_name_directive(
    job_name: str, map_count: tp.Optional[int], array_parallelism: int
) -> tp.Tuple[str, bool]:
    """Generate the job name BSUB directive.

    Returns (directive_line, is_array_job).
    """
    if map_count is not None:
        # Array job: use LSF array syntax (1-based indexing)
        array_spec = f"[1-{map_count}]"
        if array_parallelism < map_count:
            array_spec = f"[1-{map_count}]%{array_parallelism}"
        return f'#BSUB -J "{job_name}{array_spec}"', True
    return f'#BSUB -J "{job_name}"', False


def _bsub_time_directive(time_min: int) -> str:
    """Generate the wall time BSUB directive in HH:MM format."""
    hours = time_min // 60
    mins = time_min % 60
    return f"#BSUB -W {hours}:{mins:02d}"


def _bsub_resource_directives(**kwargs: tp.Any) -> tp.List[str]:
    """Generate resource-related BSUB directives.

    Accepts: nodes, cpus_per_task, gpus_per_node, mem, account, constraint, exclude, comment
    """
    lines: tp.List[str] = []
    nodes = kwargs.get("nodes")
    if nodes is not None and nodes > 1:
        lines.append(f"#BSUB -nnodes {nodes}")
    cpus_per_task = kwargs.get("cpus_per_task")
    if cpus_per_task is not None:
        lines.append(f"#BSUB -n {cpus_per_task}")
    gpus_per_node = kwargs.get("gpus_per_node")
    if gpus_per_node is not None:
        lines.append(f'#BSUB -gpu "num={gpus_per_node}"')
    mem = kwargs.get("mem")
    if mem is not None:
        lines.append(f"#BSUB -M {mem}")
    account = kwargs.get("account")
    if account is not None:
        lines.append(f"#BSUB -P {account}")
    constraint = kwargs.get("constraint")
    if constraint is not None:
        lines.append(f'#BSUB -R "{constraint}"')
    exclude = kwargs.get("exclude")
    if exclude is not None:
        lines.append(f"#BSUB -R \"select[hname!='{exclude}']\"")
    comment = kwargs.get("comment")
    if comment is not None:
        lines.append(f'#BSUB -Jd "{comment}"')
    return lines


def _bsub_warning_directives(signal_delay_s: int) -> tp.List[str]:
    """Generate warning signal BSUB directives.

    LSF uses -wa for warning action and -wt for warning time.
    Many LSF installations expect -wt in minutes (not seconds).
    Use ceil(minutes) so that e.g. 90s becomes 2 minutes.
    """
    warn_min = max(1, (signal_delay_s + 59) // 60)
    return [f"#BSUB -wt '{warn_min}'", "#BSUB -wa 'USR2'"]


def _bsub_additional_parameters_directives(
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]]
) -> tp.List[str]:
    """Generate directives from additional_parameters escape hatch."""
    if additional_parameters is None:
        return []
    lines: tp.List[str] = []
    for key, value in additional_parameters.items():
        if isinstance(value, bool) and value:
            lines.append("#BSUB -" + key)
        elif not isinstance(value, bool):
            lines.append(f"#BSUB -{key} {shlex.quote(str(value))}")
    return lines


def _bsub_env_setup_lines() -> tp.List[str]:
    """Generate environment setup lines including array job handling."""
    return [
        "",
        "# Environment setup",
        "export SUBMITIT_EXECUTOR=lsf",
        "export SUBMITIT_LSF_NTASKS=1",
        "export SUBMITIT_LSF_NNODES=1",
        "export SUBMITIT_LSF_NODEID=0",
        "export SUBMITIT_LSF_GLOBAL_RANK=0",
        "export SUBMITIT_LSF_LOCAL_RANK=0",
        "",
        "# Handle array jobs",
        # LSF sets LSB_JOBINDEX=0 for non-array jobs.
        'if [ "$LSB_JOBINDEX" != "0" ] && [ -n "$LSB_JOBINDEX" ]; then',
        '    export SUBMITIT_LSF_ARRAY_JOB_ID="$LSB_JOBID"',
        '    export SUBMITIT_LSF_ARRAY_TASK_ID="$LSB_JOBINDEX"',
        "fi",
    ]


# pylint: disable=too-many-arguments,unused-argument,too-many-locals
def _make_bsub_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitit",
    queue: tp.Optional[str] = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: tp.Optional[int] = None,
    cpus_per_task: tp.Optional[int] = None,
    gpus_per_node: tp.Optional[int] = None,
    setup: tp.Optional[tp.List[str]] = None,
    teardown: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: tp.Optional[str] = None,
    constraint: tp.Optional[str] = None,
    exclude: tp.Optional[str] = None,
    account: tp.Optional[str] = None,
    stderr_to_stdout: bool = False,
    array_parallelism: int = 256,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> str:
    """Creates the content of a bsub file with provided parameters.

    Parameters
    ----------
    See LSF bsub documentation for most parameters.

    Below are the parameters that differ from LSF documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the warning signal and the actual kill of the LSF job.
    setup: list
        a list of commands to run in bsub before running the main command
    teardown: list
        a list of commands to run in bsub after running the main command
    map_count: int
        number of jobs in the array (internal use)
    additional_parameters: dict
        Forces any parameter to a given value in bsub. This can be useful
        to add parameters which are not currently available in submitit.
        Eg: {"W": "1:00"}

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added
    """
    # Build the bsub script
    paths = utils.JobPaths(folder=folder)
    stdout = str(paths.stdout).replace("%j", "%J").replace("%t", "0")
    stderr = str(paths.stderr).replace("%j", "%J").replace("%t", "0")

    lines = ["#!/bin/bash", "", "# BSUB directives"]

    # Job name and array handling
    job_name_line, is_array = _bsub_job_name_directive(job_name, map_count, array_parallelism)
    lines.append(job_name_line)
    if is_array:
        stdout = stdout.replace("%J", "%J_%I")
        stderr = stderr.replace("%J", "%J_%I")

    # Queue
    if queue is not None:
        lines.append(f"#BSUB -q {queue}")

    # Time limit
    lines.append(_bsub_time_directive(time))

    # Resource directives
    lines.extend(
        _bsub_resource_directives(
            nodes=nodes,
            cpus_per_task=cpus_per_task,
            gpus_per_node=gpus_per_node,
            mem=mem,
            account=account,
            constraint=constraint,
            exclude=exclude,
            comment=comment,
        )
    )

    # Output files
    lines.append(f"#BSUB -o {shlex.quote(stdout)}")
    if not stderr_to_stdout:
        lines.append(f"#BSUB -e {shlex.quote(stderr)}")

    # Warning signal directives
    lines.extend(_bsub_warning_directives(signal_delay_s))

    # Additional parameters (escape hatch)
    lines.extend(_bsub_additional_parameters_directives(additional_parameters))

    # Environment setup
    lines.extend(_bsub_env_setup_lines())

    # User setup commands
    if setup is not None:
        lines += ["", "# User setup"] + setup

    # Main command
    # Run the command as a child process so the parent shell can forward the warning signal
    # (USR2 by default) to Python. This is important because `bkill -s USR2 <jobid>` targets the
    # job's main process (the shell script), and without forwarding, the shell may exit before
    # the Python handler checkpoints/requeues.
    lines += [
        "",
        "# Main command",
        f"{command} &",
        "SUBMITIT_MAIN_PID=$!",
        'trap "kill -USR2 $SUBMITIT_MAIN_PID 2>/dev/null || true" USR2',
        "wait $SUBMITIT_MAIN_PID",
    ]

    # User teardown commands
    if teardown is not None:
        lines += ["", "# User teardown"] + teardown

    lines.append("")
    return "\n".join(lines)


def _convert_mem(mem_gb: float) -> str:
    """Convert memory in GB to LSF format."""
    if mem_gb == int(mem_gb):
        return f"{int(mem_gb)}G"
    return f"{int(mem_gb * 1024)}M"
