import os
import pytz
import datetime


def get_log_file(logpath: str) -> str:
    if not logpath:
        default_logpath = os.environ.get("METHOD_LOG_DIR")
        assert default_logpath, "Source setup_env.sh"
        logpath = default_logpath
    return get_file(path=logpath, file="log", suffix="csv")


def get_file(path: str, file: str, suffix: str) -> str:
    os.makedirs(path, exist_ok=True)
    HT = pytz.timezone("Europe/Helsinki")
    dt = datetime.datetime.now().astimezone(HT)
    dt_str = dt.strftime("_%d_%m_%Y_%H_%M_%S")
    fileName = path + "/" + file + dt_str + "." + suffix
    return fileName
