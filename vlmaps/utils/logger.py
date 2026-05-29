import logging
import datetime
from enum import IntEnum
import os
from pathlib import Path


class LogLevel(IntEnum):
    NONE = -1,
    ERROR = 0,
    WARNING = 1,
    INFO = 2,
    DEBUG = 3,

def logLevelToLogging(level):
    if(level == LogLevel.ERROR):
        return logging.ERROR
    elif (level == LogLevel.WARNING):
        return logging.WARNING
    elif(level == LogLevel.INFO):
        return logging.INFO
    else:
        return logging.DEBUG

class Logger:
    def __init__(self, file: str, verbosity = LogLevel.INFO, level = LogLevel.DEBUG):
        completefile = "log/" + file + ".log"
        folder = os.path.dirname(os.path.abspath(completefile))
        Path(folder).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=completefile, level=logLevelToLogging(level))

        self.verbosity = verbosity
        self.info("Run started")

    def composeMsg(self, *args):
        dt = datetime.datetime.now()
        msg = str(dt) + ": "

        for i in range(len(args)):
            arg = args[i]
            if(isinstance(arg, tuple)):
                for item in arg:
                    msg += str(item) + " "
        return msg

    def error(self, *args):
        msg = self.composeMsg(args)
        if(int(self.verbosity) > (LogLevel.ERROR)):
            print(msg)
        logging.exception(msg)

    def warn(self, *args):
        msg = self.composeMsg(args)
        if(int(self.verbosity) > (LogLevel.ERROR)):
            print(msg)
        logging.warning(msg)

    def warning(self, *args):
        msg = self.composeMsg(args)
        if (int(self.verbosity) > (LogLevel.ERROR)):
            print(msg)
        logging.warning(msg)

    def info(self, *args):
        msg = self.composeMsg(args)
        if (int(self.verbosity) > (LogLevel.WARNING)):
            print(msg)
        logging.info(msg)

    def debug(self, *args):
        msg = self.composeMsg(args)
        if (int(self.verbosity) > (LogLevel.INFO)):
            print(msg)
        logging.debug(msg)
