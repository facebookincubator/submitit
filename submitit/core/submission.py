# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import shutil
import time
import traceback
from pathlib import Path

try:  # loading numpy before loading the pickle, to avoid unexpected interactions
    # pylint: disable=unused-import
    import numpy  # type: ignore  # noqa
except ImportError:
    pass

from . import job_environment, utils
from .logger import get_logger


def process_job(folder: Path | str, submitted_pickle: Path | str | None = None) -> None:
    """Loads a pickled job, runs it and pickles the output

    Parameter
    ---------
    folder: Path/str
        path of the folder where the job pickle will be stored (with a name containing its uuid)
    submitted_pickle: Path/str or None
        original submission to use if this job has no pickle yet; may contain array/task id placeholders

    Side-effect
    -----------
    Creates a picked output file next to the job file.
    """
    os.environ["SUBMITIT_FOLDER"] = str(folder)
    env = job_environment.JobEnvironment()
    paths = env.paths
    source_pickle = paths.submitted_pickle
    # Other ranks wait for rank zero to publish one snapshot for the whole rerun.
    if submitted_pickle is not None and env.global_rank == 0:
        source_pickle = paths._format_id(submitted_pickle)
    logger = get_logger()
    logger.info(f"Starting with {env}")
    wait_time = 60
    for _ in range(wait_time):
        # A requeued job must resume its own checkpoint, not the original submission.
        if paths.submitted_pickle.exists():
            source_pickle = paths.submitted_pickle
        if source_pickle.exists():
            break
        time.sleep(1)
    if not source_pickle.exists():
        raise RuntimeError(
            f"Waited for {wait_time} seconds but could not find submitted jobs in path:\n{source_pickle}"
        )
    logger.info(f"Loading pickle: {source_pickle}")
    paths.folder.mkdir(parents=True, exist_ok=True)
    try:
        if source_pickle != paths.submitted_pickle and env.global_rank == 0:
            with utils.temporary_save_path(paths.submitted_pickle) as tmppath:
                shutil.copyfile(source_pickle, tmppath)
            source_pickle = paths.submitted_pickle
        delayed = utils.DelayedSubmission.load(source_pickle)
        env = job_environment.JobEnvironment()
        env._handle_signals(paths, delayed)
        result = delayed.result()
        logger.info("Job completed successfully")
        del delayed  # if it blocks here, you have a race condition that must be solved!
        with utils.temporary_save_path(paths.result_pickle) as tmppath:  # save somewhere else, and move
            utils.cloudpickle_dump(("success", result), tmppath)
            del result
            logger.info("Exiting after successful completion")
    except Exception as error:  # TODO: check pickle methods for capturing traceback; pickling and raising
        try:
            with utils.temporary_save_path(paths.result_pickle) as tmppath:
                utils.cloudpickle_dump(("error", traceback.format_exc()), tmppath)
        except Exception as dumperror:
            logger.error(f"Could not dump error:\n{error}\n\nbecause of {dumperror}")
        logger.error("Submitted job triggered an exception")
        raise error


def submitit_main() -> None:
    parser = argparse.ArgumentParser(description="Run a job")
    parser.add_argument("folder", type=str, help="Folder where the jobs are stored (in subfolder)")
    parser.add_argument("--submitted-pickle", type=Path, help="Original submission to use for a new job id")
    args = parser.parse_args()
    process_job(args.folder, args.submitted_pickle)
