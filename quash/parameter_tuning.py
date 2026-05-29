import os
import sys
import argparse
from typing import Dict
from tqdm import tqdm
from methods.method import Method, SynonymType
from methods.common.tasktype import TaskType
from benchmarks.metrics import (
    get_classification,
    get_metrics,
    get_classification_detection,
)
from benchmarks.factory import Factory
from methods.common.creators.visualEncoderFactory import EncoderType
from wonderwords import RandomWord
from methods.parameter_tuning.parameter_tuner import (
    ParameterTuner,
    OptimizationObjective,
)
import matplotlib.pyplot as plt
from PIL import Image


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def run(
    method: Method,
    dataset_name: str,
    dataset_config: str,
    num: int,
    task: TaskType,
    metric: str,
    debug: bool = False,
) -> float:
    with HiddenPrints():
        dataset = Factory.get_dataset(
            dataset_name, dataset_config, task=task, shuffle=True
        )
    dataset.set_limit(num)
    our_tps: Dict[int, int] = {}
    our_fps: Dict[int, int] = {}
    our_tns: Dict[int, int] = {}
    our_fns: Dict[int, int] = {}
    our_counts: Dict[int, int] = {}

    with tqdm(total=dataset.get_max_items(), leave=False) as pbar:
        for items in dataset:
            # predict method
            our_predictions = method.predict_and_load_image_batch(items, ["other"])

            # our metrics
            assert len(our_predictions) == len(items)
            i = 0
            all_gt_boxes = []
            all_our_boxes = []
            all_gt_masks = []
            all_our_masks = []
            for i in range(len(our_predictions)):
                catname, catid, gt, image_path = items[i]
                our_mask, our_bboxes = our_predictions[i]
                all_our_boxes.append(our_bboxes)
                all_our_masks.append(our_mask)

                if task == TaskType.DETECTION:
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
                elif task == TaskType.SEGMENTATION:
                    if debug:
                        fig, ax = plt.subplots(1, 3)
                        im = Image.open(image_path)
                        ax[0].imshow(im)
                        ax[1].imshow(gt)
                        ax[2].imshow(our_mask)
                        plt.show()
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

                if catid not in our_tps:
                    our_tps[catid] = our_tp
                    our_fps[catid] = our_fp
                    our_tns[catid] = our_tn
                    our_fns[catid] = our_fn
                    our_counts[catid] = our_count
                else:
                    our_tps[catid] = our_tps[catid] + our_tp
                    our_fps[catid] = our_fps[catid] + our_fp
                    our_tns[catid] = our_tns[catid] + our_tn
                    our_fns[catid] = our_fns[catid] + our_fn
                    our_counts[catid] = our_counts[catid] + our_count
            pbar.update(1)
    our_results = get_metrics(
        our_tps,
        our_fps,
        our_tns,
        our_fns,
        our_counts,
        -100,
    )
    if metric == "f1":
        return our_results[-100].f1
    elif metric == "iou":
        return our_results[-100].iou
    else:
        raise ValueError("Unknown metric")


def new_name() -> str:
    r = RandomWord()
    adj = r.word(include_parts_of_speech=["adjectives"])
    noun = r.word(include_parts_of_speech=["noun"])
    return f"{adj}_{noun}"


def main():
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
        "--temp_folder",
        "-tf",
        dest="temp_folder",
        type=str,
        required=False,
        help="temp_folder",
        default="temp/",
    )
    parser.add_argument(
        "--task", "-t", type=int, default=TaskType.SEGMENTATION, required=False
    )
    parser.add_argument("--num", "-n", type=int, default=100, required=False)
    parser.add_argument("--num_trials", "-r", type=int, default=200, required=False)
    parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        type=str,
        required=False,
        default="coco",
    )
    parser.add_argument(
        "--dataset_config",
        "-c",
        dest="dataset_config",
        type=str,
        required=False,
        default="config/coco.yaml",
    )
    parser.add_argument(
        "--start_state",
        "-ss",
        dest="start_state",
        type=str,
        required=False,
        default="default",
    )
    parser.add_argument(
        "--experiment",
        "-x",
        dest="experiment",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--new_experiment",
        "-nx",
        dest="new_experiment",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--multiprocessing",
        "-mp",
        dest="multiprocessing",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--write_log",
        dest="write_log",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--only_synonyms",
        dest="only_synonyms",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--classifier",
        dest="classifier",
        type=str,
        required=False,
        default="cosine-svm",
    )
    parser.add_argument(
        "--metric",
        "-m",
        dest="metric",
        type=str,
        required=False,
        default="f1",
    )
    # parser.add_argument(
    #     "--classifier_config_path",
    #     dest="classifier_config_path",
    #     type=str,
    #     required=False,
    #     default="",
    # )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--prompt",
        "-p",
        dest="prompt_engineering",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument("--no_save", dest="no_save", action="store_true")
    args = parser.parse_args()
    encoder = args.encoder
    temp_folder = args.temp_folder
    num = args.num
    num_trials = args.num_trials
    dataset_name = args.dataset_name
    dataset_config = args.dataset_config
    start_state = args.start_state
    no_save = args.no_save
    experiment = args.experiment
    new_experiment = args.new_experiment
    multiprocessing = args.multiprocessing
    only_synonyms = args.only_synonyms
    objective = (
        OptimizationObjective.SYNONYMS
        if only_synonyms
        else OptimizationObjective.PARAMS
    )
    write_log = args.write_log
    classifier = args.classifier
    # classifier_config_path = args.classifier_config_path
    metric = args.metric
    debug = args.debug
    prompt_engineering = args.prompt_engineering

    # experiment name
    encoder_name = str(encoder).lower()
    if experiment and not new_experiment:
        if encoder_name not in experiment:
            experiment = f"{experiment}_img_{encoder_name}_{classifier}_{metric}"
    else:
        experiment = f"{new_name()}_img_{encoder_name}_{classifier}_{metric}"

    task = TaskType(args.task)

    tuner = ParameterTuner(
        encoder,
        classifier,
        objective,
        temp_folder,
        task,
        experiment,
        num_trials,
        multiprocessing,
        no_save,
        start_state,
        prompt_engineering,
        synonym_type=SynonymType.GENERATED,
        antonym_ratio=-1,
    )

    if not write_log:
        tuner.optimize(
            run,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            num=num,
            task=task,
            metric=metric,
            debug=debug,
        )
    tuner.write_results()


if __name__ == "__main__":
    main()
