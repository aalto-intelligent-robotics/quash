import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm
from joblib import Parallel, delayed

# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

def classify_one(i, labels, pred, gt):
    label = int(labels[i])
    prediction = (pred == label).squeeze()
    if (np.any(gt == label)):
        true = (gt == label).squeeze()
    else:
        true = np.full_like(prediction, False)

    # print(np.stack((true, prediction)).T)

    cf = classify_all(true, prediction, label)
    return cf

def classify(predictions, gt, labels, mapid, batch=False, multiprocess=0):
    output = []
    for (pred, name) in tqdm(predictions, desc="predictions", leave=False):
        classifications = []
        if(multiprocess == 0):
            for i in tqdm(range(len(labels)), desc="labels", leave=False):
                cf = classify_one(i, labels, pred, gt)
                classifications.append(cf)
        else:
            cfs = Parallel(n_jobs=multiprocess)(delayed(classify_one)(i, labels, pred, gt) for i in tqdm(range(len(labels)), desc="labels", leave=False))
            for cf in cfs:
                classifications.append(cf)
        out = aggregate(mapid, name, classifications, batch)
        output.append(out)
    return output, classifications


def instance_classification(predictions, gt_semantics, gt_instances, labels, mapid, batch=False):
    u_instances = np.unique(gt_instances)
    output = []
    for (pred, name) in predictions:
        classifications = []
        y_true = []
        y_pred = []
        for instance_id in u_instances:
            idx = (gt_instances == instance_id).squeeze()

            gu, gc = np.unique(gt_semantics[idx], return_counts=True)
            true_label = gu[np.argmax(gc)]
            # print("instance", instance_id)
            # print(np.stack((gu, gc)).T)

            preds = pred[idx]

            uniques, counts = np.unique(preds, return_counts=True)
            majority = uniques[np.argmax(counts)]

            y_true.append(true_label)
            y_pred.append(majority)

        y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred)

        uq_gt = np.unique(y_true)
        uq_p = np.unique(y_pred)
        # print(uq_gt)
        # print(uq_p)

        #! bug here
        for i in tqdm(range(len(labels)), desc="labels", leave=False):
            pred = y_pred == i
            gt = y_true == i

            tp = np.logical_and(pred == gt, gt == True)
            tn = np.logical_and(pred == gt, gt == False)
            fp = np.logical_and(pred != gt, pred == True)
            fn = np.logical_and(pred != gt, pred == False)

            disp = np.stack((y_pred, y_true, pred, gt, tp, tn, fp, fn))

            #cf = classify_one(i, labels, pred, gt)
            cf = classify_all(gt, pred, i)
            # print("="*5 + " " + str(i))
            # print(disp.T)
            # cf.print()
            classifications.append(cf)

        out = aggregate(mapid, name, classifications, batch)
        output.append(out)

    return output, classifications, (y_true, y_pred)

# ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗
# ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗
# ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝
# ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗
# ██║  ██║███████╗███████╗██║     ███████╗██║  ██║
# ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def intersection_union(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    return intersection, union


#  █████╗  ██████╗ ██████╗██╗   ██╗██████╗  █████╗  ██████╗██╗   ██╗
# ██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔══██╗██╔════╝╚██╗ ██╔╝
# ███████║██║     ██║     ██║   ██║██████╔╝███████║██║      ╚████╔╝
# ██╔══██║██║     ██║     ██║   ██║██╔══██╗██╔══██║██║       ╚██╔╝
# ██║  ██║╚██████╗╚██████╗╚██████╔╝██║  ██║██║  ██║╚██████╗   ██║
# ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝


def accuracy(classifications):
    accuracy = 0
    for classification in classifications:
        tp = classification.true_positive
        fp = classification.false_positive
        tn = classification.true_negative
        fn = classification.false_negative
        accuracy += (tp + tn) / (tp + tn + fp + fn + 1e-6)
    accuracy /= len(classifications)
    return accuracy

# ██████╗  █████╗ ██╗          █████╗  ██████╗ ██████╗██╗   ██╗██████╗  █████╗  ██████╗██╗   ██╗
# ██╔══██╗██╔══██╗██║         ██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔══██╗██╔════╝╚██╗ ██╔╝
# ██████╔╝███████║██║         ███████║██║     ██║     ██║   ██║██████╔╝███████║██║      ╚████╔╝
# ██╔══██╗██╔══██║██║         ██╔══██║██║     ██║     ██║   ██║██╔══██╗██╔══██║██║       ╚██╔╝
# ██████╔╝██║  ██║███████╗    ██║  ██║╚██████╗╚██████╗╚██████╔╝██║  ██║██║  ██║╚██████╗   ██║
# ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝


def macro_balanced_accuracy(classifications):
    bal_acc = 0
    for classification in classifications:
        tp = classification.true_positive
        tn = classification.true_negative
        fn = classification.false_negative
        fp = classification.false_positive
        temp_recall = tp / (tp + fn + 1e-6)
        temp_specificity = tn / (tn + fp + 1e-6)
        temp_bal_acc = (temp_specificity + temp_recall) / 2
        bal_acc += temp_bal_acc
    bal_acc /= len(classifications)
    return bal_acc


def micro_balanced_accuracy(classifications):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for classification in classifications:
        tp += classification.true_positive
        tn += classification.true_negative
        fn += classification.false_negative
        fp += classification.false_positive
    temp_recall = tp / (tp + fn + 1e-6)
    temp_specificity = tn / (tn + fp + 1e-6)
    balanced_accuracy = (temp_specificity + temp_recall) / 2
    return balanced_accuracy

# ██████╗ ██████╗ ███████╗ ██████╗██╗███████╗██╗ ██████╗ ███╗   ██╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔═══██╗████╗  ██║
# ██████╔╝██████╔╝█████╗  ██║     ██║███████╗██║██║   ██║██╔██╗ ██║
# ██╔═══╝ ██╔══██╗██╔══╝  ██║     ██║╚════██║██║██║   ██║██║╚██╗██║
# ██║     ██║  ██║███████╗╚██████╗██║███████║██║╚██████╔╝██║ ╚████║
# ╚═╝     ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

def macro_precision(classifications):
    precision = 0
    for classification in classifications:
        tp = classification.true_positive
        fp = classification.false_positive
        temp_precision = tp / (tp + fp + 1e-6)
        precision += temp_precision
    precision /= len(classifications)
    return precision

def micro_precision(classifications):
    tp = 0
    fp = 0
    for classification in classifications:
        tp += classification.true_positive
        fp += classification.false_positive
    precision = tp / (tp + fp + 1e-6)
    return precision

# ███╗   ██╗██████╗ ██╗   ██╗
# ████╗  ██║██╔══██╗██║   ██║
# ██╔██╗ ██║██████╔╝██║   ██║
# ██║╚██╗██║██╔═══╝ ╚██╗ ██╔╝
# ██║ ╚████║██║      ╚████╔╝
# ╚═╝  ╚═══╝╚═╝       ╚═══╝


def macro_NPV(classifications):
    NPV = 0
    for classification in classifications:
        # classification.print()
        tn = classification.true_negative
        fn = classification.false_negative
        temp_NPV = tn / (tn + fn + 1e-6)
        NPV += temp_NPV
    NPV /= len(classifications)
    return NPV


def micro_NPV(classifications):
    tn = 0
    fn = 0
    for classification in classifications:
        # classification.print()
        tn += classification.true_negative
        fn += classification.false_negative
    NPV = tn / (tn + fn + 1e-6)
    return NPV

# ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗
# ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║
# ██████╔╝█████╗  ██║     ███████║██║     ██║
# ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║
# ██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗
# ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝

def macro_recall(classifications):
    recall = 0
    for classification in classifications:
        tp = classification.true_positive
        fn = classification.false_negative
        temp_recall = tp / (tp + fn + 1e-6)
        recall += temp_recall
    recall /= len(classifications)
    return recall


def micro_recall(classifications):
    tp = 0
    fn = 0
    for classification in classifications:
        tp += classification.true_positive
        fn += classification.false_negative
    recall = tp / (tp + fn + 1e-6)
    return recall

# ███████╗██████╗ ███████╗ ██████╗██╗███████╗██╗ ██████╗██╗████████╗██╗   ██╗
# ██╔════╝██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██║╚══██╔══╝╚██╗ ██╔╝
# ███████╗██████╔╝█████╗  ██║     ██║█████╗  ██║██║     ██║   ██║    ╚████╔╝
# ╚════██║██╔═══╝ ██╔══╝  ██║     ██║██╔══╝  ██║██║     ██║   ██║     ╚██╔╝
# ███████║██║     ███████╗╚██████╗██║██║     ██║╚██████╗██║   ██║      ██║
# ╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝╚═╝     ╚═╝ ╚═════╝╚═╝   ╚═╝      ╚═╝


def macro_specificity(classifications):
    specificity = 0
    for classification in classifications:
        # classification.print()
        tn = classification.true_negative
        fp = classification.false_positive
        temp_specificity = tn / (tn + fp + 1e-6)
        specificity += temp_specificity
    specificity /= len(classifications)
    return specificity


def micro_specificity(classifications):
    tn = 0
    fp = 0
    for classification in classifications:
        # classification.print()
        tn += classification.true_negative
        fp += classification.false_positive
    specificity = tn / (tn + fp + 1e-6)
    return specificity


# ███████╗ ██╗
# ██╔════╝███║
# █████╗  ╚██║
# ██╔══╝   ██║
# ██║      ██║
# ╚═╝      ╚═╝

def macro_f1(classifications):
    f1 = 0
    for classification in classifications:
        tp = classification.true_positive
        fn = classification.false_negative
        fp = classification.false_positive
        temp_recall = tp / (tp + fn + 1e-6)
        temp_precision = tp / (tp + fp + 1e-6)
        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)
        f1 += temp_f1
    f1 /= len(classifications)
    return f1

def micro_f1(classifications):
    # P = micro_precision(classifications)
    # R = micro_recall(classifications)
    # f1 = 2*P*R / (P + R + 1e-6)
    tp = 0
    fn = 0
    fp = 0
    for classification in classifications:
        tp += classification.true_positive
        fn += classification.false_negative
        fp += classification.false_positive
    f1 = (2*tp)/(2*tp + fp + fn)
    return f1

# ██╗ ██████╗ ██╗   ██╗
# ██║██╔═══██╗██║   ██║
# ██║██║   ██║██║   ██║
# ██║██║   ██║██║   ██║
# ██║╚██████╔╝╚██████╔╝
# ╚═╝ ╚═════╝  ╚═════╝

def macro_iou(classifications):
    iou = 0
    for classification in classifications:
        i = classification.intersection
        u = classification.union
        iou += i/(u + 1e-6)
    iou /= len(classifications)
    return iou

def micro_iou(classifications):
    i = 0
    u = 0
    for classification in classifications:
        i += classification.intersection
        u += classification.union
    iou = i/(u + 1e-6)
    return iou

#  █████╗ ██╗     ██╗
# ██╔══██╗██║     ██║
# ███████║██║     ██║
# ██╔══██║██║     ██║
# ██║  ██║███████╗███████╗
# ╚═╝  ╚═╝╚══════╝╚══════╝

class Classification():
    id = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    intersection = 0
    union = 0

    def __init__(self, id, tp, fp, tn, fn, i, u):
        self.id = id
        self.true_positive = tp
        self.false_positive = fp
        self.true_negative = tn
        self.false_negative = fn
        self.intersection = i
        self.union = u
        self.count = tp + fp + tn + fn
        self.get_accuracy()
        self.get_precision()
        self.get_recall()
        self.get_f1()
        self.get_iou()
        self.get_tn_ratio()

    def get_accuracy(self) -> None:
        div = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        if div != 0:
            self.accuracy = (self.true_positive + self.true_negative) / div
        else:
            self.accuracy = 0

    def get_precision(self) -> None:
        div = self.true_positive + self.false_positive
        if div != 0:
            self.precision = self.true_positive / div
        else:
            self.precision = 0

    def get_recall(self) -> None:
        div = self.true_positive + self.false_negative
        if div != 0:
            self.recall = self.true_positive / div
        else:
            self.recall = 0

    def get_f1(self) -> None:
        div = 2 * self.true_positive + self.false_positive + self.false_negative
        if div != 0:
            self.f1 = 2 * self.true_positive / div
        else:
            self.f1 = 0

    def get_iou(self) -> None:
        div = self.true_positive + self.false_positive + self.false_negative
        if div != 0:
            self.iou = self.true_positive / div
        else:
            self.iou = 0

    def get_tn_ratio(self) -> None:
        self.all_predictions = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        div = self.all_predictions
        if div != 0:
            self.true_negative_ratio = self.true_negative / div
        else:
            self.true_negative_ratio = 0

    def __str__(self) -> str:
        out = ""
        out += f"tp:        {self.true_positive}\n"
        out += f"fp:        {self.false_positive}\n"
        out += f"tn:        {self.true_negative}\n"
        out += f"fn:        {self.false_negative}\n"
        out += f"count:     {self.count}\n"
        out += "-----------------\n"
        out += f"accuracy:  {self.accuracy:.4f}\n"
        out += f"precision: {self.precision:.4f}\n"
        out += f"recall:    {self.recall:.4f}\n"
        out += "-----------------\n"
        out += f"f1:        {self.f1:.4f}\n"
        out += f"iou:       {self.iou:.4f}\n"
        #out += f"tn_ratio:  {self.true_negative_ratio:.4f}\n"
        return out

    def print(self):
        print(self.__str__())

    def tostring(self):
        return self.__str__()

    def to_string(self):
        return self.__str__()

    def to_csv(self):
        string = ""
        string += str(self.id) + ";"
        string += str(self.true_positive) + ";"
        string += str(self.true_negative) + ";"
        string += str(self.false_positive) + ";"
        string += str(self.false_negative) + ";"
        string += str(self.intersection) + ";"
        string += str(self.union)
        return string

def classify_all(y_true, y_pred, id):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    i, u = intersection_union(y_true, y_pred)
    cf = Classification(id, tp, fp, tn, fn, i, u)
    return cf

def aggregate(map, name, classifications, batch = True):
    acc = accuracy(classifications)
    macro_averaged_precision = macro_precision(classifications)
    micro_averaged_precision = micro_precision(classifications)
    macro_averaged_recall = macro_recall(classifications)
    micro_averaged_recall = micro_recall(classifications)
    macro_averaged_f1 = macro_f1(classifications)
    micro_averaged_f1 = micro_f1(classifications)
    macro_averaged_iou = macro_iou(classifications)
    micro_averaged_iou = micro_iou(classifications)
    macro_averaged_specificity = macro_specificity(classifications)
    micro_averaged_specificity = micro_specificity(classifications)
    macro_averaged_NPV = macro_NPV(classifications)
    micro_averaged_NPV = micro_NPV(classifications)
    macro_averaged_balanced_accuracy = macro_balanced_accuracy(classifications)
    micro_averaged_balanced_accuracy = micro_balanced_accuracy(classifications)

    outitem = []
    outitem.append(map)
    outitem.append(name)
    outitem.append(acc)
    outitem.append(macro_averaged_precision)
    outitem.append(micro_averaged_precision)
    outitem.append(macro_averaged_recall)
    outitem.append(micro_averaged_recall)
    outitem.append(macro_averaged_f1)
    outitem.append(micro_averaged_f1)
    outitem.append(macro_averaged_iou)
    outitem.append(micro_averaged_iou)
    outitem.append(macro_averaged_specificity)
    outitem.append(micro_averaged_specificity)
    outitem.append(macro_averaged_NPV)
    outitem.append(micro_averaged_NPV)
    outitem.append(macro_averaged_balanced_accuracy)
    outitem.append(micro_averaged_balanced_accuracy)

    # if(not batch):
    #     print("Name:", name)
    #     print("accuracy", acc)
    #     print("macro_averaged_precision", macro_averaged_precision)
    #     print("micro_averaged_precision", micro_averaged_precision)
    #     print("macro_averaged_recall", macro_averaged_recall)
    #     print("micro_averaged_recall", micro_averaged_recall)
    #     print("macro_averaged_f1", macro_averaged_f1)
    #     print("micro_averaged_f1", micro_averaged_f1)
    #     print("macro IoU", macro_averaged_iou)
    #     print("micro IoU", micro_averaged_iou)
    #     print("macro averaged specificity", macro_averaged_specificity)
    #     print("micro averaged specificity", micro_averaged_specificity)
    #     print("macro averaged NPV", macro_averaged_NPV)
    #     print("micro averaged NPV", micro_averaged_NPV)
    #     print("macro averaged balanced accuracy", macro_averaged_balanced_accuracy)
    #     print("micro averaged balanced accuracy", micro_averaged_balanced_accuracy)

    return outitem


# ███████╗██╗  ██╗██╗     ███████╗ █████╗ ██████╗ ███╗   ██╗
# ██╔════╝██║ ██╔╝██║     ██╔════╝██╔══██╗██╔══██╗████╗  ██║
# ███████╗█████╔╝ ██║     █████╗  ███████║██████╔╝██╔██╗ ██║
# ╚════██║██╔═██╗ ██║     ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║
# ███████║██║  ██╗███████╗███████╗██║  ██║██║  ██║██║ ╚████║
# ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝


def classifyItem(gt, pred, m, name, batch):
    if(gt.size == 0):
        return None
    accuracy, macro_averaged_precision, micro_averaged_precision, macro_averaged_recall, micro_averaged_recall, macro_averaged_f1, micro_averaged_f1, iou = classification(gt, pred)
    outitem = []
    outitem.append(m)
    outitem.append(name)
    outitem.append(accuracy)
    outitem.append(macro_averaged_precision)
    outitem.append(micro_averaged_precision)
    outitem.append(macro_averaged_recall)
    outitem.append(micro_averaged_recall)
    outitem.append(macro_averaged_f1)
    outitem.append(micro_averaged_f1)
    outitem.append(iou)

    # if(not batch):
    #     print("Label:", name)
    #     print("accuracy", accuracy)
    #     print("macro_averaged_precision", macro_averaged_precision)
    #     print("micro_averaged_precision", micro_averaged_precision)
    #     print("macro_averaged_recall", macro_averaged_recall)
    #     print("micro_averaged_recall", micro_averaged_recall)
    #     print("macro_averaged_f1", macro_averaged_f1)
    #     print("micro_averaged_f1", micro_averaged_f1)
    #     print("IoU", iou)

    return outitem


def classification(y_true, y_pred):
    # correct predicions / all predictions
    # TP = true positive
    # TN = true negatice
    # FP = false positive
    # FN = false negative

    # Accuracy Score = (TP + TN) / (TP + TN + FP + FN)
    accuracy = metrics.accuracy_score(y_true, y_pred)

    # Precision = TP / (TP + FP)
    # Macro averaged precision: calculate precision for all classes individually and then average them
    macro_averaged_precision = metrics.precision_score(
        y_true, y_pred, average='macro')

    # Micro averaged precision: calculate class wise true positive and false positive and then use that to calculate overall precision
    micro_averaged_precision = metrics.precision_score(
        y_true, y_pred, average='micro')

    # Recall = TP / (TP + FN)
    macro_averaged_recall = metrics.recall_score(
        y_true, y_pred, average='macro')
    micro_averaged_recall = metrics.recall_score(
        y_true, y_pred, average='micro')

    # F1 = 2PR / (P + R)
    macro_averaged_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    micro_averaged_f1 = metrics.f1_score(y_true, y_pred, average='micro')

    # Intersect over Union (Jaccard index)
    intersection = (y_true == y_pred).sum()
    union = y_true.size + y_pred.size - intersection
    iou = intersection/union

    return accuracy, macro_averaged_precision, micro_averaged_precision, macro_averaged_recall, micro_averaged_recall, macro_averaged_f1, micro_averaged_f1, iou
