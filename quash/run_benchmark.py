import os
import sys
import signal
import argparse
from typing import Dict, List, Any, Tuple
import datetime
import warnings
import threading
# import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from tqdm import tqdm
from queue import Queue
import matplotlib.pyplot as plt
from methods.base.base import BasePredictor
from methods.baseline import Baseline
from methods.method import Method
from methods.common.factory import get_embedders
from methods.common.tasktype import TaskType
from PIL import Image
from benchmarks.metrics import (
    get_classification,
    get_metrics,
    # get_single_metric,
    get_classification_detection,
    Result,
)
from benchmarks.factory import Factory, Dataset
from methods.common.config_reader import ConfigReader
import matplotlib.patches as patches
from methods.common.creators.visualEncoderFactory import EncoderType
from wonderwords import RandomWord


# ignore deprecation warnings
warnings.simplefilter("ignore")

# this is required for partial results saving
runner = None
running = True


# this is required for partial results saving
def signal_handler(sig, frame):
    sys.exit(0)
    global running
    if not running:
        sys.exit(0)
    running = True
    global runner
    print("Interrupted, writing results...")
    runner.process_results()
    sys.exit(0)


def new_name() -> str:
    r = RandomWord()
    adj = r.word(include_parts_of_speech=["adjectives"])
    noun = r.word(include_parts_of_speech=["noun"])
    return f"{adj}_{noun}"


def dictsum(dictionary: dict) -> int:
    return int(np.sum(np.array([x for x in dictionary.values()])))


# main runner class
class BenchmarkRunner:
    def __init__(
        self,
        encoder: EncoderType,
        tempfolder: str,
        weights_path: str,
        device: str,
        network: str,
        model_name: str,
        visualize: bool,
        clean: bool,
        clean_cache: bool,
        results_folder: str,
        num: int,
        num_of_synonyms: int,
        classifier: str,
        api_model: str,
        prompt_env: str,
        dataset_config_file: str,
        temp_result_writing: int,
        cache: bool,
        measure_runtime: bool,
        run_baseline: bool,
        max_image_size: int,
        task: TaskType,
        debug_results: bool,
        precompute_classifiers: bool,
        classifier_config_path: str,
        experiment: str,
    ) -> None:
        # interrupt handler: saves partial results
        signal.signal(signal.SIGINT, signal_handler)

        # read params
        self.tempfolder = tempfolder
        self.weights_path = weights_path
        self.device = device
        self.network = network
        self.model_name = model_name
        self.visualize = visualize
        self.clean = clean
        self.clean_cache = clean_cache
        self.num = num
        self.num_of_synonyms = num_of_synonyms
        self.classifier = classifier
        self.api_model = api_model
        self.prompt_env = prompt_env
        self.dataset_config_file = dataset_config_file
        self.temp_result_writing = temp_result_writing
        self.cache = cache
        self.measure_runtime = measure_runtime
        self.run_baseline = run_baseline
        self.max_image_size = max_image_size
        self.aggregate_label = -1
        self.total_runtime: datetime.timedelta
        self.task = task
        self.debug_results = debug_results
        self.precompute_classifiers = precompute_classifiers
        self.classifier_config_path = classifier_config_path

        self.config = ConfigReader(dataset_config_file)
        dataset_name = self.config.get_config("dataset")
        assert dataset_name is not None

        if experiment:
            self.experiment_name = experiment
            self.results_folder = f"{results_folder}/{self.experiment_name}"
        else:
            name_exists = True
            while name_exists:
                self.experiment_name = f"{new_name()}_{dataset_name}_{encoder}"
                self.results_folder = f"{results_folder}/{self.experiment_name}"
                name_exists = Path(self.results_folder).exists()
        assert self.results_folder
        print(f"Start experiment {self.experiment_name}")

        # create dirs
        Path(tempfolder).mkdir(parents=True, exist_ok=True)
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        # Create dataset
        self.dataset_name: str = dataset_name
        self.absence: bool = self.dataset_name == "coco_absence"
        self.dataset: Dataset = Factory.get_dataset(
            self.dataset_name, dataset_config_file, TaskType(task)
        )
        self.dataset.set_limit(num)
        self.category_names: Dict[int, str] = self.dataset.get_categories()

        # clear downloaded images and embedding cache
        if clean_cache:
            img_dir = self.config.get_config("img_dir")
            assert img_dir
            files = [
                f
                for f in os.listdir(tempfolder)
                if os.path.isfile(os.path.join(tempfolder, f))
            ]
            for f in files:
                os.remove(os.path.join(tempfolder, f))
            files = [
                f
                for f in os.listdir(img_dir)
                if os.path.isfile(os.path.join(img_dir, f))
            ]
            for f in files:
                os.remove(os.path.join(img_dir, f))

        # create embedders
        self.image_embedder, self.text_embedder = get_embedders(
            encoder=encoder,
            weights_path=weights_path,
            device=device,
            network=network,
            model_name=model_name,
            tempfolder=tempfolder,
            cache=self.cache,
            force_resize_images=False,
            force_resize_size=None,
            max_image_size=self.max_image_size,
        )

        # create methods
        self.method: Method = Method(
            task=task,
            encoder=encoder,
            num_of_synonyms=num_of_synonyms,
            environment=prompt_env,
            classifier=classifier,
            api_model=api_model,
            measure=self.measure_runtime,
            temp_folder=tempfolder,
            max_retries=int(1e2),
            classifier_config_path=self.classifier_config_path,
        )
        self.method.set_embedders(self.image_embedder, self.text_embedder)
        if self.run_baseline:
            self.baseline: Baseline = Baseline(measure=self.measure_runtime)
            self.baseline.set_embedders(self.image_embedder, self.text_embedder)

        # precompute classifiers
        if self.precompute_classifiers:
            for id in tqdm(self.category_names, desc="Precompute classifiers"):
                cat = self.category_names[id]
                self.method.create_predictor(cat, ["other"])

        # setup run variables
        self.task_count: int = 0
        self.baseline_tps: Dict[int, int] = {}
        self.baseline_fps: Dict[int, int] = {}
        self.baseline_tns: Dict[int, int] = {}
        self.baseline_fns: Dict[int, int] = {}
        self.baseline_counts: Dict[int, int] = {}
        self.our_tps: Dict[int, int] = {}
        self.our_fps: Dict[int, int] = {}
        self.our_tns: Dict[int, int] = {}
        self.our_fns: Dict[int, int] = {}
        self.our_counts: Dict[int, int] = {}

    def classify_one(self, items: List[Tuple[Any, ...]]) -> None:
        our_predictions = self.method.predict_and_load_image_batch(items, ["other"])

        # our metrics
        assert len(our_predictions) == len(items)
        i = 0
        for i in range(len(our_predictions)):
            catname, catid, gt, image_path = items[i]
            our_mask, our_bboxes = our_predictions[i]

            if self.task == TaskType.DETECTION:
                gt_comp_boxes = []
                for bb in gt:
                    x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                    gt_comp_boxes.append([x1, y1, x2, y2])
                (
                    our_tp,
                    our_fp,
                    our_tn,
                    our_fn,
                    our_count,
                ) = get_classification_detection(gt_comp_boxes, our_bboxes)
            elif self.task == TaskType.SEGMENTATION:
                our_prediction = our_mask.reshape(-1)
                gt_mask = gt.reshape(-1)
                assert our_prediction.shape == gt_mask.shape
                (our_tp, our_fp, our_tn, our_fn, our_count) = get_classification(
                    gt_mask, our_prediction
                )
            else:
                raise ValueError("Unknown task type")

            with self.lock:
                if catid not in self.our_tps:
                    self.our_tps[catid] = our_tp
                    self.our_fps[catid] = our_fp
                    self.our_tns[catid] = our_tn
                    self.our_fns[catid] = our_fn
                    self.our_counts[catid] = our_count
                else:
                    self.our_tps[catid] = self.our_tps[catid] + our_tp
                    self.our_fps[catid] = self.our_fps[catid] + our_fp
                    self.our_tns[catid] = self.our_tns[catid] + our_tn
                    self.our_fns[catid] = self.our_fns[catid] + our_fn
                    self.our_counts[catid] = self.our_counts[catid] + our_count

    def update_results(self, results_queue: Queue):
        """A separate thread function that aggregates results."""
        current = 0
        while True:
            result = results_queue.get()
            if result is None:
                break  # Exit signal
            catid, our_tp, our_fp, our_tn, our_fn, our_count = result

            # Update the shared dictionaries with the result
            if catid not in self.our_tps:
                self.our_tps[catid] = our_tp
                self.our_fps[catid] = our_fp
                self.our_tns[catid] = our_tn
                self.our_fns[catid] = our_fn
                self.our_counts[catid] = our_count
            else:
                self.our_tps[catid] += our_tp
                self.our_fps[catid] += our_fp
                self.our_tns[catid] += our_tn
                self.our_fns[catid] += our_fn
                self.our_counts[catid] += our_count

            # self.progress(current)
            current += 1

            if self.debug_results:
                tmp_scnt = dictsum(self.our_counts)
                tmp_stps = dictsum(self.our_tps)
                tmp_sfps = dictsum(self.our_fps)
                tmp_stns = dictsum(self.our_tns)
                tmp_sfns = dictsum(self.our_fns)
                print(
                    "{0}: {1}, {2}, {3}, {4}".format(
                        tmp_scnt, tmp_stps, tmp_stns, tmp_sfps, tmp_sfns
                    )
                )
        print("writer exiting")

    # def run_mp(self, max_workers=4) -> None:
    #     self.looping_start = datetime.datetime.now()
    #     self.lock = threading.Lock()  # To ensure thread-safe updates
    #     semaphore = threading.Semaphore(
    #         max_workers
    #     )  # Control the number of concurrent tasks

    #     with tqdm(
    #         total=self.dataset.get_max_items()
    #     ) as pbar, concurrent.futures.ThreadPoolExecutor(
    #         max_workers=max_workers
    #     ) as executor:
    #         futures = []

    #         for item in self.dataset:  # Progressively load items from the dataset
    #             semaphore.acquire()  # Limit the number of concurrent executions

    #             # Submit a task to the executor
    #             future = executor.submit(self.classify_one, item)
    #             futures.append(future)

    #             # When a task is done, release the semaphore
    #             future.add_done_callback(lambda f: semaphore.release())

    #             # Handle progress bar updates
    #             future.add_done_callback(lambda f: pbar.update(1))

    #         # Wait for all futures to complete
    #         for future in concurrent.futures.as_completed(futures):
    #             future.result()  # Ensure exceptions are raised, if any
    #             self.task_count += 1

    #     self.looping_end = datetime.datetime.now()
    #     self.total_runtime = self.looping_end - self.looping_start
    #     self.process_results()

    def run_mp(self, max_workers=4) -> None:
        self.looping_start = datetime.datetime.now()
        total = self.dataset.get_max_items()
        self.lock = threading.Lock()

        # Stream tasks; no futures list, no callbacks
        with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(
            total=total
        ) as pbar:
            for _ in ex.map(
                self.classify_one, self.dataset, chunksize=1
            ):  # yields as tasks complete (in order)
                pbar.update(1)
                self.task_count += 1

        self.looping_end = datetime.datetime.now()
        self.total_runtime = self.looping_end - self.looping_start
        self.process_results()

    # ██████╗ ██╗   ██╗███╗   ██╗
    # ██╔══██╗██║   ██║████╗  ██║
    # ██████╔╝██║   ██║██╔██╗ ██║
    # ██╔══██╗██║   ██║██║╚██╗██║
    # ██║  ██║╚██████╔╝██║ ╚████║
    # ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

    def run(self) -> None:
        self.looping_start = datetime.datetime.now()
        with tqdm(total=self.dataset.get_max_items()) as pbar:
            for items in self.dataset:
                # predict method
                our_predictions = self.method.predict_and_load_image_batch(
                    items, ["other"]
                )

                # our metrics
                assert len(our_predictions) == len(items)
                i = 0
                all_gt_boxes = []
                all_our_boxes = []
                all_gt_masks = []
                all_our_masks = []
                all_baseline_masks = []
                for i in range(len(our_predictions)):
                    catname, catid, gt, image_path = items[i]
                    our_mask, our_bboxes = our_predictions[i]
                    all_our_boxes.append(our_bboxes)
                    all_our_masks.append(our_mask)

                    if self.task == TaskType.DETECTION:
                        gt_comp_boxes = []
                        for bb in gt:
                            x1, x2, y1, y2 = [
                                bb[0],
                                bb[0] + bb[2],
                                bb[1],
                                bb[1] + bb[3],
                            ]
                            gt_comp_boxes.append([x1, y1, x2, y2])
                        all_gt_boxes.append(gt_comp_boxes)
                        (
                            our_tp,
                            our_fp,
                            our_tn,
                            our_fn,
                            our_count,
                        ) = get_classification_detection(gt_comp_boxes, our_bboxes)
                    elif self.task == TaskType.SEGMENTATION:
                        our_prediction = our_mask.reshape(-1)
                        gt_mask = gt.reshape(-1)
                        assert our_prediction.shape == gt_mask.shape
                        (
                            our_tp,
                            our_fp,
                            our_tn,
                            our_fn,
                            our_count,
                        ) = get_classification(gt_mask, our_prediction)
                        all_gt_masks.append(gt_mask)

                    else:
                        raise ValueError("Unknown task type")

                    if catid not in self.our_tps:
                        self.our_tps[catid] = our_tp
                        self.our_fps[catid] = our_fp
                        self.our_tns[catid] = our_tn
                        self.our_fns[catid] = our_fn
                        self.our_counts[catid] = our_count
                    else:
                        self.our_tps[catid] = self.our_tps[catid] + our_tp
                        self.our_fps[catid] = self.our_fps[catid] + our_fp
                        self.our_tns[catid] = self.our_tns[catid] + our_tn
                        self.our_fns[catid] = self.our_fns[catid] + our_fn
                        self.our_counts[catid] = self.our_counts[catid] + our_count

                    # FIXME: debug
                    # temp_results = get_single_metric(
                    #     -1,
                    #     our_tp,
                    #     our_fp,
                    #     our_tn,
                    #     our_fn,
                    #     our_count,
                    # )
                    # if temp_results.accuracy > 0 and temp_results.precision == 0:
                    #     print("Alert!")
                    #     fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
                    #     fig.suptitle(catname)
                    #     image = Image.open(image_path)
                    #     ax11.imshow(image)
                    #     ax11.set_title("orig")
                    #     ax12.imshow(gt)
                    #     ax12.set_title("gt")
                    #     ax21.imshow(all_our_masks[i])
                    #     ax21.set_title("ours")
                    #     plt.show()

                # predict baseline
                if self.run_baseline:
                    for item in items:
                        catname, catid, gt, image_path = item
                        (
                            baseline_mask,
                            baseline_bboxes,
                            _,
                            _,
                            _,
                        ) = self.baseline.predict_and_load_image(
                            image_path, catname, ["other"]
                        )
                        all_baseline_masks.append(baseline_mask)
                        if self.task == TaskType.DETECTION:
                            gt_comp_boxes = []
                            for bb in gt:
                                x1, x2, y1, y2 = [
                                    bb[0],
                                    bb[0] + bb[2],
                                    bb[1],
                                    bb[1] + bb[3],
                                ]
                                gt_comp_boxes.append([x1, y1, x2, y2])
                            (
                                baseline_tp,
                                baseline_fp,
                                baseline_tn,
                                baseline_fn,
                                baseline_count,
                            ) = get_classification_detection(
                                gt_comp_boxes, baseline_bboxes
                            )
                        elif self.task == TaskType.SEGMENTATION:
                            baseline_prediction = baseline_mask.reshape(-1)
                            gt_mask = gt.reshape(-1)
                            assert baseline_prediction.shape == gt_mask.shape

                            # baseline metrics
                            (
                                baseline_tp,
                                baseline_fp,
                                baseline_tn,
                                baseline_fn,
                                baseline_count,
                            ) = get_classification(gt_mask, baseline_prediction)

                        else:
                            raise ValueError("Unknown task type")

                        if catid not in self.baseline_tps:
                            self.baseline_tps[catid] = baseline_tp
                            self.baseline_fps[catid] = baseline_fp
                            self.baseline_tns[catid] = baseline_tn
                            self.baseline_fns[catid] = baseline_fn
                            self.baseline_counts[catid] = baseline_count

                        else:
                            self.baseline_tps[catid] = (
                                self.baseline_tps[catid] + baseline_tp
                            )
                            self.baseline_fps[catid] = (
                                self.baseline_fps[catid] + baseline_fp
                            )
                            self.baseline_tns[catid] = (
                                self.baseline_tns[catid] + baseline_tn
                            )
                            self.baseline_fns[catid] = (
                                self.baseline_fns[catid] + baseline_fn
                            )
                            self.baseline_counts[catid] = (
                                self.baseline_counts[catid] + baseline_count
                            )

                self.task_count += 1

                # debug results view
                if self.debug_results:
                    tmp_scnt = dictsum(self.our_counts)
                    tmp_stps = dictsum(self.our_tps)
                    tmp_sfps = dictsum(self.our_fps)
                    tmp_stns = dictsum(self.our_tns)
                    tmp_sfns = dictsum(self.our_fns)
                    print(
                        "{0}: {1}, {2}, {3}, {4}".format(
                            tmp_scnt, tmp_stps, tmp_stns, tmp_sfps, tmp_sfns
                        )
                    )

                # visualization
                if self.visualize:
                    for i, item in enumerate(items):
                        catname, catid, gt, image_path = item
                        image = Image.open(image_path)
                        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
                        fig.suptitle(catname)
                        ax11.imshow(image)
                        ax11.set_title("orig")

                        if self.task == TaskType.DETECTION:
                            ax12.imshow(image)
                            ax12.set_title("gt")
                            # BOUNDING BOX DRAWING
                            for bb in all_gt_boxes[i]:
                                # Create a rectangle patch
                                # rect = patches.Rectangle(
                                #     (bb[0], bb[1]),
                                #     bb[2],
                                #     bb[3],
                                #     linewidth=2,
                                #     edgecolor="r",
                                #     facecolor="none",
                                # )

                                x_min, y_min, x_max, y_max = bb
                                width = x_max - x_min
                                height = y_max - y_min
                                # Create a rectangle patch
                                rect = patches.Rectangle(
                                    (x_min, y_min),
                                    width,
                                    height,
                                    linewidth=2,
                                    edgecolor="r",
                                    facecolor="none",
                                )

                                # Add the rectangle to the Axes
                                ax12.add_patch(rect)

                            ax21.imshow(all_our_masks[i])
                            ax21.set_title("ours")
                            # BOUNDING BOX DRAWING
                            for bb in all_our_boxes[i]:
                                x_min, y_min, x_max, y_max = bb
                                width = x_max - x_min
                                height = y_max - y_min
                                # Create a rectangle patch
                                rect = patches.Rectangle(
                                    (x_min, y_min),
                                    width,
                                    height,
                                    linewidth=2,
                                    edgecolor="r",
                                    facecolor="none",
                                )
                                # Add the rectangle to the Axes
                                ax21.add_patch(rect)
                        elif self.task == TaskType.SEGMENTATION:
                            ax12.imshow(gt)
                            ax12.set_title("gt")
                            ax21.imshow(all_our_masks[i])
                            ax21.set_title("ours")
                            if self.run_baseline:
                                ax22.imshow(all_baseline_masks[i])
                                ax22.set_title("baseline")
                        plt.show()

                # clean images
                if self.clean:
                    for i, item in enumerate(items):
                        catname, catid, gt, image_path = item
                        os.remove(image_path)

                # intermediate result saving
                if (
                    self.temp_result_writing > 0
                    and self.task_count % self.temp_result_writing == 0
                ):
                    suffix = "tmp-" + str(self.task_count)
                    self.looping_end = datetime.datetime.now()
                    self.total_runtime = self.looping_end - self.looping_start
                    self.get_results()
                    self.write_results_to_file(suffix, intermediate=True)
                    print("Intermediate results saved")

                # self.dataset.print_progress()
                if self.dataset.update_progressbar():  # When starting a new sublist
                    pbar.update(1)

        self.looping_end = datetime.datetime.now()
        self.total_runtime = self.looping_end - self.looping_start

        # write results to file
        self.process_results()

    # ███████╗███╗   ██╗██████╗
    # ██╔════╝████╗  ██║██╔══██╗
    # █████╗  ██╔██╗ ██║██║  ██║
    # ██╔══╝  ██║╚██╗██║██║  ██║
    # ███████╗██║ ╚████║██████╔╝
    # ╚══════╝╚═╝  ╚═══╝╚═════╝

    def process_results(self) -> None:
        self.get_results()
        self.display_results()
        self.write_results_to_file()

    def get_results(self) -> None:
        self.our_results = get_metrics(
            self.our_tps,
            self.our_fps,
            self.our_tns,
            self.our_fns,
            self.our_counts,
            self.aggregate_label,
        )
        if self.run_baseline:
            self.baseline_results = get_metrics(
                self.baseline_tps,
                self.baseline_fps,
                self.baseline_tns,
                self.baseline_fns,
                self.baseline_counts,
                self.aggregate_label,
            )

    def write_results_to_file(
        self, suffix: str = "", intermediate: bool = False
    ) -> None:
        folder = (
            f"{self.results_folder}/intermediate"
            if intermediate
            else self.results_folder
        )

        our_result_path = self.write_results(
            self.method,
            self.total_runtime,
            "our",
            self.dataset_name,
            self.our_results,
            folder,
            self.task_count,
            self.category_names,
            self.aggregate_label,
            self.absence,
            suffix,
        )

        baseline_result_path = ""
        if self.run_baseline:
            baseline_result_path = self.write_results(
                self.baseline,
                self.total_runtime,
                "baseline",
                self.dataset_name,
                self.baseline_results,
                folder,
                self.task_count,
                self.category_names,
                self.aggregate_label,
                self.absence,
                suffix,
            )

        absence_filepath = ""
        if self.absence:
            absence_queries = self.dataset.get_queries()
            absence_filepath = self.getFile(
                self.results_folder, "absence_queries", "csv"
            )
            with open(absence_filepath, "w") as f:
                for orig, new in absence_queries:
                    print(orig, "->", new)
                    f.write(orig + " -> " + new + "\n")

        # notification email
        try:
            if self.run_baseline and baseline_result_path:
                attachments = [
                    our_result_path,
                    baseline_result_path,
                    self.method.logfile,
                ]
            else:
                attachments = [
                    our_result_path,
                    self.method.logfile,
                ]

            if self.absence and absence_filepath:
                attachments.append(absence_filepath)
            outfile = os.environ.get("RUNFILE")
            assert outfile, "source setup_env.sh"
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(" ".join(attachments))
        except Exception as e:
            print(e)

    def display_results(self) -> None:
        print(" ")

        self.print_results(
            "our",
            self.our_results,
            self.task_count,
            self.category_names,
            self.aggregate_label,
            self.absence,
        )

        if self.run_baseline:
            self.print_results(
                "baseline",
                self.baseline_results,
                self.task_count,
                self.category_names,
                self.aggregate_label,
                self.absence,
            )

    def plot(
        self,
        r: int,
        c: int,
        matrix: np.ndarray,
        title: str,
        idx: int,
        absolute: bool = False,
        vmin: float = 0,
        vmax: float = 1,
    ) -> None:
        plt.subplot(r, c, idx)
        if absolute:
            plt.imshow(matrix, vmin=vmin, vmax=vmax, cmap="seismic")
        else:
            plt.imshow(matrix, cmap="seismic")
        plt.title(title)
        plt.colorbar()

    def print_results(
        self,
        title: str,
        results: Dict[int, Result],
        task_count: int,
        category_names: Dict[int, str],
        aggregate_label: int,
        absence: bool = False,
    ) -> None:
        title = title[0].upper() + title[1:]
        print(title, "results over", task_count, "classifications")
        for id in results:
            result = results[id]
            if id == aggregate_label:
                name = "Aggregate"
            else:
                try:
                    name = category_names[result.id]
                except Exception:
                    name = "error"
            print("Category", result.id, name)
            print("Number of predictions", result.count)
            if absence:
                print("TN      ", result.tn)
                print("TN ratio", result.tn_ratio)
            else:
                print("Accuracy ", result.accuracy)
                print("Precision", result.precision)
                print("Recall   ", result.recall)
                print("F1       ", result.f1)
                print("IoU      ", result.iou)
            print("")

    def addField(self, line: str, value) -> str:
        line += str(value) + ";"
        return line

    def getFile(self, path: str, file: str, suffix: str) -> str:
        os.makedirs(path, exist_ok=True)
        fileName = path + "/" + file + "." + suffix
        file_exists = os.path.exists(fileName)
        if file_exists:
            dt = datetime.datetime.now()
            dt_str = dt.strftime("_%d_%m_%Y_%H_%M_%S")
            fileName = path + "/" + file + dt_str + "." + suffix
        return fileName

    def write_results(
        self,
        model: BasePredictor,
        runtime: datetime.timedelta,
        title: str,
        dataset: str,
        results: Dict[int, Result],
        dir: str,
        task_count: int,
        category_names: Dict[int, str],
        aggregate_label: int,
        absence: bool = False,
        suffix: str = "",
    ) -> str:
        # write metadata
        filename = title + "_" + dataset + "_metadata"
        if suffix:
            filename += "_" + suffix
        filepath = self.getFile(dir, filename, "txt")
        with open(filepath, "w") as f:
            f.write("total runtime: " + str(runtime) + "\n")
            f.write(model.get_method_description() + "\n")
        print(f"Metadata written to: {dir}/{filename}")

        # write results
        filename = title + "_" + dataset + "_results"
        if suffix:
            filename += "_" + suffix
        if absence:
            filename += "_absence"
        suffix = "csv"
        filepath = self.getFile(dir, filename, suffix)
        with open(filepath, "w") as f:
            # f.write("total runtime: " + str(runtime) + "\n")
            # f.write(model.get_method_description() + "\n")
            if absence:
                f.write("method;cat_id;cat_name;num_task;num_pred;tn;tn_ratio\n")
            else:
                f.write(
                    "method;cat_id;cat_name;num_task;"
                    "num_pred;accuracy;precision;recall;f1;iou\n"
                )
            for id in results:
                result = results[id]
                if id == aggregate_label:
                    name = "Aggregate"
                else:
                    try:
                        name = category_names[result.id]
                    except Exception:
                        name = "error"
                line = ""
                line = self.addField(line, title)
                line = self.addField(line, id)
                line = self.addField(line, name)
                line = self.addField(line, task_count)
                line = self.addField(line, result.count)
                if absence:
                    line = self.addField(line, result.tn)
                    line = self.addField(line, result.tn_ratio)
                else:
                    line = self.addField(line, result.accuracy)
                    line = self.addField(line, result.precision)
                    line = self.addField(line, result.recall)
                    line = self.addField(line, result.f1)
                    line = self.addField(line, result.iou)
                f.write(line.rstrip(";") + "\n")
        print(f"Results written to: {dir}/{filename}")
        return filepath

    def strfdelta(self, tdelta: datetime.timedelta, fmt: str) -> str:
        d = {"days": str(tdelta.days)}
        hours, rem = divmod(tdelta.seconds, 3600)
        minutes, secs = divmod(rem, 60)
        d["hours"] = str(hours)
        d["minutes"], d["seconds"] = str(minutes), str(secs)

        d["days"] = f"{d['days']:02d}"
        d["hours"] = f"{d['hours']:02d}"
        d["minutes"] = f"{d['minutes']:02d}"
        d["seconds"] = f"{d['seconds']:02d}"

        return fmt.format(**d)

    def print_progress(
        self, looping_start: datetime.datetime, count: int, maxid: int
    ) -> None:
        dataset_progress = self.dataset.get_progress_str()
        runtime = self.predict_running_time(looping_start, count, maxid)
        queries, cached = self.method.get_num_queries()
        query_p = queries / (queries + cached) * 100
        tnow = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        if runtime is None:
            runtime = "-"
        print(
            tnow,
            "-",
            dataset_progress,
            "-",
            runtime,
            "-",
            "Queries:",
            f"{queries:5d}",
            ",",
            f"{query_p:6.2f}%",
        )

    def predict_running_time(
        self, looping_start: datetime.datetime, count: int, maxid: int
    ) -> str:
        loop_end = datetime.datetime.now()
        delta = loop_end - looping_start
        dt = delta.total_seconds()
        if count == 0:
            return ""
        t_per_one = dt / count
        remaining = maxid - count
        p_total = t_per_one * remaining
        dt_total = datetime.timedelta(seconds=p_total)
        done = loop_end + datetime.timedelta(seconds=p_total)
        strdone = done.strftime("%d-%m-%Y %H:%M")

        runtime_str = "Remaining: "
        runtime_str += self.strfdelta(
            dt_total, "{days}d, {hours}h, {minutes}m, {seconds}s"
        )
        runtime_str += " Done: "
        runtime_str += strdone
        return runtime_str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        "-e",
        dest="encoder",
        type=EncoderType.from_string,
        required=False,
        default=EncoderType.LSEG,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to the model weights.",
        default="weights/demo_e200.ckpt",
    )
    parser.add_argument(
        "--device",
        "-dv",
        dest="device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        required=False,
        help="Use cpu or cuda.",
    )
    parser.add_argument(
        "--network",
        "-nw",
        dest="network",
        type=str,
        choices=["clip", "flava", "tbd"],
        required=False,
        help="Which NN model to use",
        default="clip",
    )
    parser.add_argument(
        "--model_name",
        "-md",
        dest="model_name",
        type=str,
        required=False,
        help="Path to model",
        default="ViT-B/32",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        dest="visualize",
        action="store_true",
        required=False,
        help="visualize",
        default=False,
    )
    parser.add_argument(
        "--clean",
        "-cl",
        dest="clean",
        action="store_true",
        required=False,
        help="clean",
        default=False,
    )
    parser.add_argument(
        "--clean_cache",
        "-cc",
        dest="clean_cache",
        action="store_true",
        required=False,
        help="clean_cache",
        default=False,
    )
    parser.add_argument(
        "--temp_folder",
        "-tf",
        dest="temp_folder",
        type=str,
        required=False,
        help="temp_folder",
        default="temp/",
    )
    parser.add_argument(
        "--results_folder",
        "-rf",
        dest="results_folder",
        type=str,
        required=False,
        help="results_folder",
        default="results/",
    )
    parser.add_argument(
        "--num",
        "-n",
        dest="num",
        type=int,
        required=False,
        help="num",
        default=-1,
    )
    parser.add_argument("--num_of_synonyms", "-ns", type=int, default=21)
    parser.add_argument(
        "--prompt_env", "-pe", type=str, default="world", required=False
    )
    parser.add_argument(
        "--classifier", "-clf", type=str, default="cosine-svm", required=False
    )
    parser.add_argument(
        "--api_model", "-am", type=str, default="gpt4-turbo", required=False
    )
    parser.add_argument(
        "--dataset_config",
        "-c",
        type=str,
        default="config/coco.yaml",
        required=False,
    )
    parser.add_argument(
        "--temp_result_writing", "-tr", type=int, default=10000, required=False
    )
    parser.add_argument(
        "--cache", "-ch", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--measure_runtime", "-mr", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--run_baseline", "-rb", action="store_true", default=False, required=False
    )
    parser.add_argument("--max_image_size", "-ms", type=int, default=-1, required=False)
    parser.add_argument(
        "--task", "-t", type=TaskType, default=TaskType.SEGMENTATION, required=False
    )
    parser.add_argument(
        "--debug_results", "-dr", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--precompute_classifiers",
        "-pcc",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--multiprocessing", "-mp", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--multiprocessing_workers", "-mpw", type=int, default=4, required=False
    )
    parser.add_argument(
        "--classifier_config_path", "-cfg", type=str, default="", required=False
    )
    parser.add_argument("--experiment", "-x", type=str, default="", required=False)
    args = parser.parse_args()

    tempfolder = args.temp_folder
    weights_path = args.weights_path
    device = args.device
    network = args.network
    model_name = args.model_name
    visualize = args.visualize
    clean = args.clean
    clean_cache = args.clean_cache
    results_folder = args.results_folder
    num = args.num
    num_of_synonyms = args.num_of_synonyms
    classifier = args.classifier
    api_model = args.api_model
    prompt_env = args.prompt_env
    dataset_config_file = args.dataset_config
    temp_result_writing = args.temp_result_writing
    cache = args.cache
    measure_runtime = args.measure_runtime
    run_baseline = args.run_baseline
    max_image_size = args.max_image_size
    task = TaskType(args.task)
    debug_results = args.debug_results
    precompute_classifiers = args.precompute_classifiers
    multiprocessing = args.multiprocessing
    multiprocessing_workers = args.multiprocessing_workers
    encoder = args.encoder
    classifier_config_path = args.classifier_config_path
    experiment = args.experiment

    Path(Path(weights_path).parent).mkdir(exist_ok=True, parents=True)

    # this is for the interrupt handler - saves partial results
    global runner
    runner = BenchmarkRunner(
        encoder,
        tempfolder,
        weights_path,
        device,
        network,
        model_name,
        visualize,
        clean,
        clean_cache,
        results_folder,
        num,
        num_of_synonyms,
        classifier,
        api_model,
        prompt_env,
        dataset_config_file,
        temp_result_writing,
        cache,
        measure_runtime,
        run_baseline,
        max_image_size,
        task,
        debug_results,
        precompute_classifiers,
        classifier_config_path,
        experiment,
    )

    if multiprocessing:
        runner.run_mp(multiprocessing_workers)
    else:
        runner.run()


if __name__ == "__main__":
    main()
