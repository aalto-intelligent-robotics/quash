import numpy as np
from sklearn import metrics
from typing import Tuple, List


class Result:
    def __init__(self, id) -> None:
        self.id = id

    def get(self, tp: int, fp: int, tn: int, fn: int, count: int) -> None:
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.count = count
        self.get_accuracy()
        self.get_precision()
        self.get_recall()
        self.get_f1()
        self.get_iou()
        self.get_tn_ratio()

    def get_accuracy(self) -> None:
        div = self.tp + self.fp + self.tn + self.fn
        if div != 0:
            self.accuracy = (self.tp + self.tn) / div
        else:
            self.accuracy = 0

    # micro averaged
    def get_precision(self) -> None:
        div = self.tp + self.fp
        if div != 0:
            self.precision = self.tp / div
        else:
            self.precision = 0

    # micro averaged
    def get_recall(self) -> None:
        div = self.tp + self.fn
        if div != 0:
            self.recall = self.tp / div
        else:
            self.recall = 0

    # micro averaged
    def get_f1(self) -> None:
        div = 2 * self.tp + self.fp + self.fn
        if div != 0:
            self.f1 = 2 * self.tp / div
        else:
            self.f1 = 0

    # micro averaged
    def get_iou(self) -> None:
        div = self.tp + self.fp + self.fn
        if div != 0:
            self.iou = self.tp / div
        else:
            self.iou = 0

    def get_tn_ratio(self) -> None:
        self.all_predictions = self.tp + self.fp + self.tn + self.fn
        div = self.all_predictions
        if div != 0:
            self.tn_ratio = self.tn / div
        else:
            self.tn_ratio = 0

    def __str__(self) -> str:
        out = ""
        out += f"tp: {self.tp}\n"
        out += f"fp: {self.fp}\n"
        out += f"tn: {self.tn}\n"
        out += f"fn: {self.fn}\n"
        out += f"count: {self.count}\n"
        out += f"accuracy: {self.accuracy}\n"
        out += f"precision: {self.precision}\n"
        out += f"recall: {self.recall}\n"
        out += f"f1: {self.f1}\n"
        out += f"iou: {self.iou}\n"
        out += f"tn_ratio: {self.tn_ratio}\n"
        return out


def get_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int, int]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    count = np.sum((tp, fp, tn, fn))
    return tp, fp, tn, fn, count


def calculate_iou(box1, box2):
    """
    Helper function to calculate the Intersection over Union (IoU)
    between two bounding boxes.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate intersection coordinates
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Intersection area
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Box areas
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Union area
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# FIXME: we are not finding instances, rather areas. So we will fail this test
def get_classification_detection(
    true_bb: List[List[int]], pred_bb: List[List[int]], iou_threshold: float = 0.5
) -> Tuple[int, int, int, int, int]:
    """
    Calculate True Positives (TP), False Positives (FP), True Negatives (TN),
    and False Negatives (FN) for bounding boxes.

    Parameters:
        true_bb (List): Ground truth bounding boxes [x_min, y_min, x_max, y_max].
        pred_bb (List): Predicted bounding boxes [x_min, y_min, x_max, y_max].
        iou_threshold (float): IoU threshold to classify TP, FP, and FN.

    Returns:
        Tuple: (TP, FP, TN, FN)
    """

    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    tn = 0  # True Negative (note: TN may not be useful for object detection)

    matched_true_boxes = []  # Keep track of which true boxes were matched

    # Loop through each predicted bounding box
    for pred_box in pred_bb:
        best_iou = 0
        best_gt_idx = -1
        # Find the best matching ground truth box
        for idx, gt_box in enumerate(true_bb):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # If the best IoU is above the threshold, it's a True Positive
        if best_iou >= iou_threshold and best_gt_idx not in matched_true_boxes:
            tp += 1
            matched_true_boxes.append(best_gt_idx)
        else:
            # Otherwise, it's a False Positive
            fp += 1

    # All ground truth boxes not matched are False Negatives
    fn = len(true_bb) - len(matched_true_boxes)

    # True negatives are typically not counted in object detection
    # because there's no explicit measure of "no object" in this context
    tn = 0  # This is just a placeholder, not typically used for object detection tasks.

    # print(tp, fp, tn, fn)
    count = np.sum((tp, fp, tn, fn))
    return tp, fp, tn, fn, count


def get_metrics(
    tps: dict, fps: dict, tns: dict, fns: dict, counts: dict, aggregate_label: int
) -> dict:
    results = {}
    agg_tp = 0
    agg_fp = 0
    agg_tn = 0
    agg_fn = 0
    agg_count = 0
    for id in tps:
        tp = tps[id]
        fp = fps[id]
        tn = tns[id]
        fn = fns[id]
        count = counts[id]
        result = get_single_metric(id, tp, fp, tn, fn, count)
        results[id] = result
        agg_tp += tp
        agg_fp += fp
        agg_tn += tn
        agg_fn += fn
        agg_count += count
    result = Result(aggregate_label)
    result.get(agg_tp, agg_fp, agg_tn, agg_fn, agg_count)
    results[aggregate_label] = result
    return results


def get_single_metric(
    id: int, tp: int, fp: int, tn: int, fn: int, count: int
) -> Result:
    result = Result(id)
    result.get(tp, fp, tn, fn, count)
    return result


def classify(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    # correct predicions / all predictions
    # TP = true positive
    # TN = true negatice
    # FP = false positive
    # FN = false negative

    # Accuracy Score = (TP + TN) / (TP + TN + FP + FN)
    accuracy = metrics.accuracy_score(y_true, y_pred)

    # Precision = TP / (TP + FP)
    # Macro averaged precision: calculate precision for all classes
    # individually and then average them
    macro_averaged_precision = metrics.precision_score(y_true, y_pred, average="macro")

    # Micro averaged precision: calculate class wise true positive and
    # false positive and then use that to calculate overall precision
    micro_averaged_precision = metrics.precision_score(y_true, y_pred, average="micro")

    # Recall = TP / (TP + FN)
    macro_averaged_recall = metrics.recall_score(y_true, y_pred, average="macro")
    micro_averaged_recall = metrics.recall_score(y_true, y_pred, average="micro")

    # F1 = 2PR / (P + R)
    macro_averaged_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    micro_averaged_f1 = metrics.f1_score(y_true, y_pred, average="micro")

    # Intersect over Union (Jaccard index)
    macro_averaged_iou = metrics.jaccard_score(y_true, y_pred, average="macro")
    micro_averaged_iou = metrics.jaccard_score(y_true, y_pred, average="micro")

    return (
        float(accuracy),
        float(macro_averaged_precision),
        float(micro_averaged_precision),
        float(macro_averaged_recall),
        float(micro_averaged_recall),
        float(macro_averaged_f1),
        float(micro_averaged_f1),
        float(macro_averaged_iou),
        float(micro_averaged_iou),
    )
