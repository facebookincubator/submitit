# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time
import traceback
from pathlib import Path
from typing import Union

try:  # loading numpy before loading the pickle, to avoid unexpected interactions
    # pylint: disable=unused-import
    import numpy  # type: ignore  # noqa
except ImportError:
    pass

from . import job_environment, utils
from .logger import get_logger


def process_job(folder: Union[Path, str]) -> None:
    """Loads a pickled job, runs it and pickles the output

    Parameter
    ---------
    folder: Path/str
        path of the folder where the job pickle will be stored (with a name containing its uuid)

    Side-effect
    -----------
    Creates a picked output file next to the job file.
    """
    folder = Path(folder)
    env = job_environment.JobEnvironment()
    paths = utils.JobPaths(folder, job_id=env.job_id, task_id=env.global_rank)
    logger = get_logger()
    logger.info(f"Starting with {env}")
    logger.info(f"Loading pickle: {paths.submitted_pickle}")
    wait_time = 60
    for _ in range(wait_time):
        if not paths.submitted_pickle.exists():
            time.sleep(1)
    if not paths.submitted_pickle.exists():
        raise RuntimeError(
            f"Waited for {wait_time} seconds but could not find submitted jobs in path:\n{paths.submitted_pickle}"
        )
    try:
        delayed = utils.DelayedSubmission.load(paths.submitted_pickle)
        env = job_environment.JobEnvironment()
        env._handle_signals(paths, delayed)
        result = delayed.result()
        with utils.temporary_save_path(paths.result_pickle) as tmppath:  # save somewhere else, and move
            utils.cloudpickle_dump(("success", result), tmppath)
            logger.info("Job completed successfully")
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
    args = parser.parse_args()
    process_job(args.folder)
