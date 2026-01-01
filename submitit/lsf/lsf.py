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

    # pylint: disable=too-many-branches
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

    def _interrupt(self, timeout: bool = False) -> None:  # pylint: disable=unused-argument
        """Sends preemption or timeout signal to the job (for testing purpose)

        Parameter
        ---------
        timeout: bool
            Whether to trigger a job time-out (if False, it triggers preemption)
        """
        cmd = ["bkill", "-s", LsfJobEnvironment.USR_SIG, self.job_id]
        subprocess.check_call(cmd)


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


# pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches,too-many-statements
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
    nonlsf = [
        "nonlsf",
        "folder",
        "command",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "setup",
        "teardown",
        "signal_delay_s",
        "stderr_to_stdout",
    ]
    parameters = {k: v for k, v in locals().items() if v is not None and k not in nonlsf}

    # Build the bsub script
    paths = utils.JobPaths(folder=folder)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)

    # Convert submitit path placeholders to LSF placeholders
    # submitit uses %j for job id and %t for task id
    # LSF uses %J for job id and %I for array index
    stdout = stdout.replace("%j", "%J").replace("%t", "0")
    stderr = stderr.replace("%j", "%J").replace("%t", "0")

    lines = ["#!/bin/bash", "", "# BSUB directives"]

    # Job name
    if "job_name" in parameters:
        job_name_val = parameters.pop("job_name")
        if map_count is not None:
            # Array job: use LSF array syntax (1-based indexing)
            array_spec = f"[1-{map_count}]"
            if array_parallelism < map_count:
                array_spec = f"[1-{map_count}]%{array_parallelism}"
            lines.append(f'#BSUB -J "{job_name_val}{array_spec}"')
            # For arrays, include %I in output paths
            stdout = stdout.replace("%J", "%J_%I")
            stderr = stderr.replace("%J", "%J_%I")
        else:
            lines.append(f'#BSUB -J "{job_name_val}"')

    # Queue
    if "queue" in parameters:
        lines.append(f'#BSUB -q {parameters.pop("queue")}')

    # Time limit (LSF uses -W for wall time in minutes or HH:MM format)
    if "time" in parameters:
        time_min = parameters.pop("time")
        hours = time_min // 60
        mins = time_min % 60
        lines.append(f"#BSUB -W {hours}:{mins:02d}")

    # Nodes
    if "nodes" in parameters:
        n = parameters.pop("nodes")
        if n > 1:
            lines.append(f"#BSUB -nnodes {n}")

    # CPUs per task
    if "cpus_per_task" in parameters:
        lines.append(f'#BSUB -n {parameters.pop("cpus_per_task")}')

    # GPUs per node
    if "gpus_per_node" in parameters:
        gpu_count = parameters.pop("gpus_per_node")
        # LSF GPU syntax varies by installation, common format:
        lines.append(f'#BSUB -gpu "num={gpu_count}"')

    # Memory
    if "mem" in parameters:
        lines.append(f'#BSUB -M {parameters.pop("mem")}')

    # Account/project
    if "account" in parameters:
        lines.append(f'#BSUB -P {parameters.pop("account")}')

    # Constraint (LSF uses -R for resource requirements)
    if "constraint" in parameters:
        lines.append(f'#BSUB -R "{parameters.pop("constraint")}"')

    # Exclude hosts
    if "exclude" in parameters:
        lines.append(f'#BSUB -R "select[hname!=\'{parameters.pop("exclude")}\']"')

    # Comment
    if "comment" in parameters:
        lines.append(f'#BSUB -Jd "{parameters.pop("comment")}"')

    # Output files
    lines.append(f"#BSUB -o {shlex.quote(stdout)}")
    if not stderr_to_stdout:
        lines.append(f"#BSUB -e {shlex.quote(stderr)}")

    # Signal handling for checkpointing
    # LSF uses -wa for warning action and -wt for warning time
    # Many LSF installations expect -wt in minutes (and don't accept a seconds suffix like '90s').
    # Use ceil(minutes) so that e.g. 90s becomes 2 minutes.
    warn_min = max(1, (signal_delay_s + 59) // 60)
    lines.append("#BSUB -wt '" + str(warn_min) + "'")
    lines.append("#BSUB -wa 'USR2'")

    # Additional parameters (escape hatch)
    if additional_parameters is not None:
        for key, value in additional_parameters.items():
            if isinstance(value, bool):
                if value:
                    lines.append("#BSUB -" + key)
            else:
                lines.append(f"#BSUB -{key} {shlex.quote(str(value))}")

    # Handle remaining parameters
    for key in list(parameters.keys()):
        if key == "ntasks_per_node":
            # LSF doesn't have a direct equivalent; handled via -n
            parameters.pop(key)
            continue

    # Environment setup
    lines += [
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

    # User setup commands
    if setup is not None:
        lines += ["", "# User setup"] + setup

    # Main command
    lines += [
        "",
        "# Main command",
        command,
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
