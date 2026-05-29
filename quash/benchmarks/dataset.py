import threading
import time
from abc import ABC, abstractmethod
import typing
from typing import List, Tuple, Any, Optional
import ast
import datetime
from methods.common.config_reader import ConfigReader
from methods.common.tasktype import TaskType


class Dataset(ABC):
    def __init__(
        self,
        config_file: str,
        task: TaskType,
        shuffle: bool = False,
        fetch_size: int = 10,
        buffer_size: int = 100,
    ) -> None:
        self.config = ConfigReader(config_file)  # Configuration for the dataset
        self.task = task
        self.shuffle = shuffle
        # List to hold the lists of sub-items (sublists)
        self.items: List[List[Tuple[Any, ...]]] = []
        self.fetch_size = fetch_size  # Number of sublists to fetch at a time
        self.buffer_size = buffer_size  # Max number of sublists to hold in memory
        self.sublist_index = 0  # Index for the current sublist
        self.subitem_index = 0  # Index for the current sub-item within a sublist
        self.max_items: int  # To be set by the child class
        self.no_more_items = False  # Flag to indicate if loading is finished
        self.load_thread = threading.Thread(
            target=self._load_items
        )  # Background thread to load items
        self.load_thread.daemon = (
            True  # Ensure the thread exits when the main program exits
        )
        self.threading_started = False
        self.ignored_classes: List[Tuple[int, str]] = []
        self.waiting_for_more = False
        self.get_lists = False
        self.limit: Optional[int] = None

        # DBG
        self.iter = 0
        self.item_iter = 0
        self.empty_lists = 0
        self.index = 0

    @abstractmethod
    def _create_item(self, overall_index) -> List[Tuple[Any, ...]]:
        """Abstract method to create sub-items for each top-level item."""
        pass

    def _load_items(self):
        """Thread to continuously load items into self.items."""
        self.index = 0

        while not self.no_more_items:
            if self.waiting_for_more or (len(self.items) < self.buffer_size):
                # Fetch a batch of sublists
                for _ in range(self.fetch_size):
                    if (self.max_items and self.index >= self.max_items) or (
                        self.limit and self.index >= self.limit
                    ):
                        self.no_more_items = True
                        break
                    new_item = self._create_item(
                        self.index
                    )  # Child class creates sublist
                    self.items.append(new_item)
                    self.index += 1
            time.sleep(0.01)  # Prevent CPU overuse

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple[Any, ...]]:
        new_item = None
        while not new_item:
            if (self.max_items and self.index >= self.max_items) or (
                self.limit and self.index >= self.limit
            ):
                raise StopIteration
            new_item = self._create_item(self.index)
            self.index += 1
            self.iter += 1

        return new_item

    def HARD__next__(self) -> List[Tuple[Any, ...]]:
        if not self.threading_started:
            self.threading_started = True
            self.looping_start = datetime.datetime.now()
            self.load_thread.start()

        if self.get_lists:
            # Check if there are items to return or if we need to wait for more items
            while self.sublist_index >= len(self.items):
                if self.no_more_items and self.sublist_index >= len(self.items):
                    # print("stop. no more items",
                    # self.sublist_index, len(self.items), self.index)
                    raise StopIteration  # All items are loaded and iterated
                # print("waiting for more 1...")
                self.waiting_for_more = True
                time.sleep(0.5)  # Wait for more items to be loaded
            self.waiting_for_more = False

            if self.sublist_index > 10:
                del self.items[0]
                self.sublist_index -= 1
            current_sublist = self.items[self.sublist_index]
            return current_sublist
        else:
            # Check if there are items to return or if we need to wait for more items
            while self.sublist_index >= len(self.items):
                if self.no_more_items and self.sublist_index >= len(self.items):
                    # print("stop. no more items",
                    # self.sublist_index, len(self.items), self.index)
                    raise StopIteration  # All items are loaded and iterated
                # print("waiting for more 1...")
                self.waiting_for_more = True
                time.sleep(0.5)  # Wait for more items to be loaded
            self.waiting_for_more = False

            # Get the current sublist
            current_sublist = self.items[self.sublist_index]

            # Handle the case where the current sublist is empty
            while len(current_sublist) == 0:
                self.empty_lists += 1
                # If no more items, raise StopIteration
                if self.no_more_items and self.sublist_index >= len(self.items):
                    # print("stop2. no more items",
                    # self.sublist_index, len(self.items), self.index)
                    raise StopIteration

                # Move to the next sublist if the current one is empty
                self.sublist_index += 1
                if self.sublist_index >= len(self.items):
                    # print("waiting for more 2...")
                    self.waiting_for_more = True
                    time.sleep(0.5)  # Wait for more items to be loaded
                else:
                    current_sublist = self.items[self.sublist_index]
                    self.waiting_for_more = False

            # Get the current sub-item from the sublist
            current_item = current_sublist[self.subitem_index]

            # Move to the next subitem or sublist
            self.subitem_index += 1
            if self.subitem_index >= len(current_sublist):
                self.subitem_index = 0
                self.sublist_index += 1
                self.item_iter += 1

                # Clean up the fully processed sublist to free memory
                if (
                    self.sublist_index > 1
                ):  # Only remove if we have moved beyond the first sublist
                    del self.items[0]  # Remove the first item, adjust index accordingly
                    self.sublist_index -= (
                        1  # Adjust the sublist index since we've removed an item
                    )

            self.iter += 1

            # print(self.iter, self.item_iter, len(self.items),
            # self.sublist_index, self.subitem_index, self.index)
            return [current_item]

    # ██╗   ██╗████████╗██╗██╗     ███████╗
    # ██║   ██║╚══██╔══╝██║██║     ██╔════╝
    # ██║   ██║   ██║   ██║██║     ███████╗
    # ██║   ██║   ██║   ██║██║     ╚════██║
    # ╚██████╔╝   ██║   ██║███████╗███████║
    # ╚═════╝    ╚═╝   ╚═╝╚══════╝╚══════╝

    # @abstractmethod
    # def get(self) -> typing.Tuple[bool, np.ndarray, str, str]:
    #     pass

    @abstractmethod
    def get_categories(self) -> typing.Dict[int, str]:
        pass

    def set_used_classes(self) -> None:
        self.used_classes_file = self.config.get_config("used_classes")
        assert self.used_classes_file
        if self.used_classes_file == "all":
            self.used_classes = None
        else:
            with open(self.used_classes_file, "r", encoding="utf-8") as f:
                file_content = f.read()
                self.used_classes = ast.literal_eval(file_content)

    def check_class(self, name: str) -> bool:
        if self.used_classes is None:
            return True
        else:
            if name not in self.used_classes:
                return False
            else:
                return True

    def set_limit(self, limit: int) -> None:
        if limit > 0:
            self.limit = limit
        else:
            self.limit = None

    # @abstractmethod
    # def get_idx(self) -> int:
    #     pass

    # @abstractmethod
    # def get_maxidx(self) -> int:
    #     pass

    @abstractmethod
    def get_progress_str(self) -> str:
        pass

    def get_max_items(self) -> int:
        if self.limit and self.limit < self.max_items:
            return self.limit
        else:
            return self.max_items

    def update_progressbar(self) -> bool:
        return self.subitem_index == 0

    def print_progress(self) -> None:
        max_items = self.get_max_items()
        cur_items = self.item_iter  # Items processed so far
        delta_t = datetime.datetime.now() - self.looping_start

        if cur_items > 0:
            # Calculate the average time per item
            avg_time_per_item = delta_t / cur_items

            # Estimate remaining time
            remaining_items = max_items - cur_items
            remaining_time = avg_time_per_item * remaining_items
            remaining_time_str = str(remaining_time)
        else:
            remaining_time_str = (
                "Calculating..."  # Handle the case where no items are processed yet
            )

        # Print progress with estimated remaining time
        print(
            "{}: {}/{} - est: {}".format(
                delta_t, cur_items, max_items, remaining_time_str
            )
        )

    # def resize_image(self, img: Image) -> Image:
    #     width, height = img.size

    #     # If neither dimension exceeds self.max_size, return the image as is
    #     if max(width, height) <= self.max_size:
    #         return img

    #     # Calculate the new dimensions while preserving the aspect ratio
    #     if width > height:
    #         new_width = self.max_size
    #         new_height = int(self.max_size * height / width)
    #     else:
    #         new_height = self.max_size
    #         new_width = int(self.max_size * width / height)

    #     # Resize the image while maintaining the aspect ratio
    #     resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    #     return resized_img
