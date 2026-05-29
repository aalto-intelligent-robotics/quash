from typing import List, Callable, Optional, Any
from os import listdir
from os.path import isfile, join, isdir
import re


def match_filter(string: str, filter: str = "") -> bool:
    if not filter:
        return True
    match = re.search(filter, string)
    return match is not None


def get_files(folder: str, filter: str = "") -> List[str]:
    return get_content(folder, isfile, filter)


def get_folders(folder: str, filter: str = "") -> List[str]:
    return get_content(folder, isdir, filter)


def _always_true(_: Any) -> bool:
    return True


def get_content(
    folder: str, func: Optional[Callable] = None, filter: str = ""
) -> List[str]:
    if func is None:
        func = _always_true
    return [
        f for f in listdir(folder) if func(join(folder, f)) and match_filter(f, filter)
    ]
