import time
import argparse
from methods.common.cache_manager import CacheManager, OverwriteOption, CleaningStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, required=True)
    parser.add_argument("--max_size", "-s", type=float, required=True)
    parser.add_argument("--unit", "-u", type=str, required=False, default="GiB")
    args = parser.parse_args()
    folder = args.dir
    max_size = args.max_size
    unit = args.unit

    assert max_size > 0, "Max size must be positive"

    mgr = CacheManager(
        folder=folder,
        max_size=max_size,
        unit=unit,
        overwrite=OverwriteOption.NEVER,
        cleaning_strategy=CleaningStrategy.LARGEST,
        clean_on_exit=False,
        clean_on_start=False,
    )

    while True:
        mgr.manage_storage()
        mgr.update_state()
        mgr.print()
        time.sleep(0.1)


if __name__ == "__main__":
    main()

