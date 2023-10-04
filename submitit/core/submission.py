# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Union

try:  # loading numpy before loading the pickle, to avoid unexpected interactions
    # pylint: disable=unused-import
    import numpy  # type: ignore  # noqa
except ImportError:
    pass

from . import job_environment, tblib, utils
from .logger import get_logger

logger = get_logger()


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
    os.environ["SUBMITIT_FOLDER"] = str(folder)
    env = job_environment.JobEnvironment()
    paths = env.paths
    logger.info(f"Starting with {env}")
    logger.info(f"Loading pickle: {paths.submitted_pickle}")
    wait_time = 60
    for _ in range(wait_time):
        if paths.submitted_pickle.exists():
            break
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
        logger.info("Job computed its result")
        # if it blocks here, you have a race condition that must be solved!
        del delayed
    except Exception as error:
        logger.error("Submitted job triggered an exception")
        with utils.temporary_save_path(paths.result_pickle) as tmp_path:
            save_error(error, tmp_path)
        raise
    except BaseException:
        logger.exception("Submitted job encoutered a system error. Will result in an UncompletedJobError")
        raise

    with utils.temporary_save_path(paths.result_pickle) as tmp_path:
        save_result(result, tmp_path)
        # if it blocks here, you have a race condition that must be solved!
        del result
        logger.info("Exitting after successful completion")


def save_result(result, tmp_path: Path):
    try:
        utils.cloudpickle_dump(("success", result), tmp_path)
        logger.info("Job completed successfully")
    except Exception as pickle_error:
        logger.error(f"Could not pickle job result because of {pickle_error}")
        save_error(pickle_error, tmp_path)


def save_error(error: Exception, tmp_path: Path) -> None:
    """Pickle the full exception with its trace using tblib."""
    try:
        # tblib needs to be installed after we have created the exception class
        # they recommend doing it just before pickling the exception.
        # This seems to be a limitation of copyreg.
        tblib.install(error)
        utils.cloudpickle_dump(("error", error), tmp_path)
    except Exception as pickle_error:
        logger.error(f"Could not pickle exception:\n{error}\n\nbecause of {pickle_error}")
        # Fallbacks to only pickling the trace
        try:
            utils.cloudpickle_dump(("error", traceback.format_exc()), tmp_path)
        except Exception as dumperror:
            logger.error(f"Could not dump exception:\n{error}\n\nbecause of {dumperror}")
            logger.error("This will trigger a JobResultsNotFoundError")
            raise


def submitit_main() -> None:
    parser = argparse.ArgumentParser(description="Run a job")
    parser.add_argument("folder", type=str, help="Folder where the jobs are stored (in subfolder)")
    args = parser.parse_args()
    process_job(args.folder)
