import os
import sys
from typing import Optional, Tuple
from pathlib import Path
from enum import Enum
import numpy as np
import shutil
from abc import abstractmethod
import pytz
import datetime

class OverwriteOption(Enum):
    NEVER = 0
    ALWAYS = 1
    DIFFERENT_SIZE = 2
    DIFFERENT_CONTENT = 3


class CleaningStrategy(Enum):
    OLDEST = 1
    LARGEST = 2


class Unit(Enum):
    MB = 0
    MiB = 1
    GB = 2
    GiB = 3

    @abstractmethod
    def from_string(unit: str):
        if unit.lower() == "mb":
            return Unit.MB
        elif unit.lower() == "mib":
            return Unit.MiB
        elif unit.lower() == "gb":
            return Unit.GB
        elif unit.lower() == "gib":
            return Unit.GiB
        else:
            raise ValueError("Unknown unit")

    def to_string(self) -> str:
        if self == Unit.MB:
            return "MB"
        elif self == Unit.MiB:
            return "MiB"
        elif self == Unit.GB:
            return "GB"
        elif self == Unit.GiB:
            return "GiB"

class CacheManager:
    def __init__(
        self,
        folder: str,
        max_size: float = -1.0,
        unit: str = "GiB",
        overwrite: OverwriteOption = OverwriteOption.NEVER,
        cleaning_strategy: CleaningStrategy = CleaningStrategy.OLDEST,
        clean_on_exit: bool = False,
        clean_on_start: bool = False,
        cache_sizes: bool = False
    ):
        # parameters
        self.folder = folder
        self.overwrite = overwrite
        self.cleaning_strategy = cleaning_strategy
        self.clean_on_start = clean_on_start
        self.clean_on_exit = clean_on_exit
        self.unit = Unit.from_string(unit)
        self.max_size, self.max_size_unit = self.convert_unit_reverse(max_size)
        self.cache_sizes = cache_sizes

        # variables
        self.total_size = 0
        self.total_files = 0
        self.num_loads = 0
        self.num_saves = 0
        self.num_overwrites = 0
        self.num_removes = 0
        self.num_rejected = 0
        self.size_cache = {}

        # create folder
        self.path = Path(self.folder)
        self.path.mkdir(parents=True, exist_ok=True)

        # clean
        if self.clean_on_start:
            self._clean()

    def __del__(self):
        if self.clean_on_exit:
            self._clean()

    def load(self, filename: str) -> Optional[np.ndarray]:
        self.num_loads += 1
        path = self._get_path(filename)
        if Path(path).exists():
            return np.load(path)
        else:
            return None

    def save(self, filename: str, data: np.ndarray) -> None:
        filepath = self._get_path(filename)
        write, exists = self._check_overwrite(filename, data)
        if write:
            new_size = sys.getsizeof(data)
            if new_size > self.max_size:
                self.num_rejected += 1
                return

            self.total_files += 1
            self.total_size += new_size
            self.num_saves += 1
            self.manage_storage()
            np.save(filepath, data, allow_pickle=True)
            if exists:
                self.num_overwrites += 1

    def convert_unit(self, value: float):
        if self.unit == Unit.MB:
            return value / (10**6), "MB"
        elif self.unit == Unit.MiB:
            return value / (2**20), "MiB"
        elif self.unit == Unit.GB:
            return value / (10**9), "GB"
        elif self.unit == Unit.GiB:
            return value / (2**30), "GiB"

    def convert_unit_reverse(self, value: float):
        if self.unit == Unit.MB:
            return value * (10**6), "MB"
        elif self.unit == Unit.MiB:
            return value * (2**20), "MiB"
        elif self.unit == Unit.GB:
            return value * (10**9), "GB"
        elif self.unit == Unit.GiB:
            return value * (2**30), "GiB"

    def print(self) -> None:
        path = Path(self.folder).resolve()
        size, count = self.total_size, self.total_files
        total, used, free = shutil.disk_usage(path)
        tp, tp_unit = self.convert_unit(total)
        up, up_unit = self.convert_unit(used)
        fp, fp_unit = self.convert_unit(free)
        used_p = up/tp*100
        size_v, size_unit = self.convert_unit(size)
        max_size_v, max_size_unit = self.convert_unit(self.max_size)
        HT = pytz.timezone("Europe/Helsinki")
        dt = datetime.datetime.now().astimezone(HT)
        dt_str = dt.strftime("%d.%m.%Y %H:%M:%S")
        print("=" * 80)
        print("CacheManager state:")
        print("-" * 40)
        print(dt_str)
        print("-" * 40)
        print(f"Folder        : {path}")
        print(f"Data size     : {size_v:.2f} {size_unit}")
        print(f"Max size      : {max_size_v:.2f} {max_size_unit}")
        print(f"  % of free   : {max_size_v/fp*100:.2f} %")
        print(f"Usage         : {self.get_usage() * 100:.2f} %")
        print(f"Num of files  : {count}")
        print(f"Num loads     : {self.num_loads}")
        print(f"Num saves     : {self.num_saves}")
        print(f"Num overwrites: {self.num_overwrites}")
        print(f"Num removes   : {self.num_removes}")
        print(f"Num rejected  : {self.num_rejected}")
        print(" ")

        print("Disk state:")
        print("-" * 40)
        print(f"Total: {tp:.2f} {tp_unit}")
        print(f"Used : {up:.2f} {up_unit}")
        print(f"Used : {used_p:.2f} %")
        print(f"Free : {fp:.2f} {fp_unit}")
        print("=" * 80)
        print(" ")

    def get_usage(self) -> float:
        return self.total_size / self.max_size

    def _check_overwrite(self, filename: str, data: np.ndarray) -> Tuple[bool, bool]:
        filepath = self._get_path(filename)
        exists = Path(filepath).exists()
        write = True
        if exists:
            if self.overwrite == OverwriteOption.NEVER:
                write = False
            elif self.overwrite == OverwriteOption.ALWAYS:
                write = True
            elif self.overwrite == OverwriteOption.DIFFERENT_SIZE:
                existing_size = Path(filepath).stat().st_size
                new_size = sys.getsizeof(data)
                write = new_size != existing_size
            elif self.overwrite == OverwriteOption.DIFFERENT_CONTENT:
                old_data = self.load(filename)
                assert old_data is not None
                write = not np.array_equal(old_data, data)
        return write, exists

    def _get_path(self, filename: str) -> str:
        if ".npy" != Path(filename).suffix:
            filename = f"{filename}.npy"
        return f"{self.folder}/{filename}"

    # def _get_size(self):
    #     size = 0
    #     count = 0
    #     for f in self.path.glob("**/*"):
    #         if f.is_file():
    #             if f in self.size_cache:
    #                 size = self.size_cache[f]
    #             else:
    #                 filesize = f.stat().st_size
    #                 size += filesize
    #                 if self.cache_sizes:
    #                     self.size_cache[f] = filesize
    #             count += 1
    #     return size, count

    def _get_size(self):
        total_size = 0
        count = 0
        cache = self.size_cache
        use_cache = getattr(self, "cache_sizes", False)

        stack = [self.path]  # start dir(s)
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue

                        # only files (no symlinks)
                        if not entry.is_file(follow_symlinks=False):
                            continue

                        key = entry.path  # use str key to avoid Path overhead
                        if key in cache:
                            sz = cache[key]
                        else:
                            try:
                                sz = entry.stat(follow_symlinks=False).st_size
                            except OSError:
                                # permission/race—skip gracefully
                                continue
                            if use_cache:
                                cache[key] = sz

                        total_size += sz     # <-- was 'size = cache[f]' bug in original
                        count += 1
            except OSError:
                # e.g., permission denied opening a directory
                continue

        return total_size, count

    def _clean(self):
        for f in self.path.glob("**/*"):
            if f.is_file():
                os.remove(f)

    def _remove(self, file: Path) -> None:
        self.total_files -= 1
        self.total_size -= file.stat().st_size
        self.num_removes += 1
        os.remove(file)

    def update_state(self) -> None:
        size, count = self._get_size()
        self.total_size = size
        self.total_files = count

    def manage_storage(self) -> bool:
        start_time = datetime.datetime.now()
        if self.max_size > 0:
            raw_ages = []
            paths = []
            raw_sizes = []
            for f in self.path.glob("**/*"):
                if f.is_file():
                    stat = f.stat()
                    age = stat.st_mtime
                    size = stat.st_size
                    raw_ages.append(age)
                    raw_sizes.append(size)
                    paths.append(f)
            while self.total_size > self.max_size:
                now = datetime.datetime.now()

                if now - start_time > datetime.timedelta(seconds=10):
                    print("Clean time limit exceeded")
                    break

                if self.cleaning_strategy == CleaningStrategy.OLDEST:
                    ages = np.array(raw_ages)
                    rmidx = np.argmin(ages)
                    self._remove(paths[rmidx])
                    del paths[rmidx]
                    del raw_ages[rmidx]
                elif self.cleaning_strategy == CleaningStrategy.LARGEST:
                    sizes = np.array(raw_sizes)
                    rmidx = np.argmax(sizes)
                    self._remove(paths[rmidx])
                    del paths[rmidx]
                    del raw_sizes[rmidx]
                else:
                    raise ValueError("Unknown cleaning strategy")
                print(f"Cleaning... {self.total_size/self.max_size*100:.2f}% used. Cleaning time {now-start_time}")
        print("")
        return True
