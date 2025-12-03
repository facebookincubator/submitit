# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging.config
import os
from typing import Union

# provide a way to change level through SUBMITTHEM_LOG_LEVEL environment variable:
# level "CRITICAL" (50) or more (eg.: "100") will deactivate submitthem logger
# "NOCONFIG" will avoid configuration
LOG_VARNAME = "SUBMITTHEM_LOG_LEVEL"
level_str = os.environ.get(LOG_VARNAME, "INFO").upper()
level: Union[int, str] = level_str if not level_str.isdigit() else int(level_str)


CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"submitthem_basic": {"format": "%(name)s %(levelname)s (%(asctime)s) - %(message)s"}},
    "handlers": {
        "submitthem_out": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "submitthem_basic",
            "stream": "ext://sys.stdout",
        },
        "submitthem_err": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "submitthem_basic",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {"submitthem": {"handlers": ["submitthem_err", "submitthem_out"], "level": level}},
}


if level != "NOCONFIG":
    logging.config.dictConfig(CONFIG)


def get_logger() -> logging.Logger:
    return logging.getLogger("submitthem")


def exception(*args: str) -> None:
    get_logger().exception(*args)


def warning(*args: str) -> None:
    get_logger().warning(*args)
