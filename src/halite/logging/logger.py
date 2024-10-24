# shamelessly took from detectron2
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/logger.py

import functools
import inspect
import logging
import os
import pprint
import sys
from typing import List, Optional

try:
    from termcolor import colored

except ImportError:
    colored = None

try:
    from rich.logging import RichHandler

except ImportError:
    RichHandler = None


class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def wrap_log_record_factory(factory):
    def wrapper(
        name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs
    ):
        if not isinstance(msg, str):
            msg = pprint.pformat(msg)

        return factory(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

    return wrapper


class DistributedLogger:
    def __init__(self, name, mesh=None, mode="rich", abbrev_name=None, keywords=None):
        self.logger = make_logger(
            name, mode=mode, abbrev_name=abbrev_name, keywords=keywords
        )
        self.mesh = mesh

    @staticmethod
    def get_call_info():
        stack = inspect.stack()

        fn = stack[3][1]
        ln = stack[3][2]
        func = stack[3][3]

        return os.path.basename(fn), ln, func

    def log(
        self,
        level,
        message,
        mesh_dim="global",
        ranks: Optional[List[int]] = None,
    ):
        if ranks is None:
            return getattr(self.logger, level)(message)

        if mesh_dim == "global":
            local_rank = self.mesh.get_rank()

        else:
            local_rank = self.mesh.get_local_rank(mesh_dim)

        if local_rank in ranks:
            return getattr(self.logger, level)(message)

    def message_prefix(self):
        return "FROM {}:{} {}()".format(*self.get_call_info())

    def info(
        self,
        message: str,
        mesh_dim="global",
        ranks: Optional[List[int]] = (0,),
    ):
        self.log("info", self.message_prefix(), mesh_dim, ranks)
        self.log("info", message, mesh_dim, ranks)

    def warning(
        self,
        message: str,
        mesh_dim="global",
        ranks: Optional[List[int]] = (0,),
    ):
        self.log("warning", self.message_prefix(), mesh_dim, ranks)
        self.log("warning", message, mesh_dim, ranks)

    def debug(
        self,
        message: str,
        mesh_dim="global",
        ranks: Optional[List[int]] = (0,),
    ):
        self.log("debug", self.message_prefix(), mesh_dim, ranks)
        self.log("debug", message, mesh_dim, ranks)

    def error(
        self,
        message: str,
        mesh_dim="global",
        ranks: Optional[List[int]] = (0,),
    ):
        self.log("error", self.message_prefix(), mesh_dim, ranks)
        self.log("error", message, mesh_dim, ranks)


@functools.lru_cache  # so that calling setup_logger multiple times won't add many handlers
def make_logger(name="main", mode="rich", abbrev_name=None, keywords=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """

    logging.setLogRecordFactory(wrap_log_record_factory(logging.getLogRecordFactory()))

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    if mode == "rich" and RichHandler is None:
        mode = "color"

    if mode == "color" and colored is None:
        mode = "plain"

    if mode == "color":
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(abbrev_name),
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    elif mode == "rich":
        logger.addHandler(
            RichHandler(
                level=logging.DEBUG, log_time_format="%m/%d %H:%M:%S", keywords=keywords
            )
        )

    elif mode == "plain":
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
