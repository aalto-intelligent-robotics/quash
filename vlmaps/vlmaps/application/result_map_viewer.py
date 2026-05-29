import os
import argparse
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import open3d as o3d
import utils.common as common
from pathlib import Path
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, MaxNLocator


def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray, name: str = "Default name"):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    _ = o3d.visualization.draw_geometries([voxel_grid], window_name=name)

def visualize_confusion_matrix(confusion_matrix, c, names):
    data = np.copy(confusion_matrix)

    normalize = c == 2
    log = c == 3
    if normalize:
        row_sums = np.sum(data, axis=1)[:, np.newaxis]
        n_mask = (row_sums > 0).squeeze()
        normalized = data[n_mask, :] / (row_sums[n_mask, :] + 1e-6)
        data[n_mask, :] = normalized

    if not normalize:
        mask_zeros = data == 0
        annot = np.where(mask_zeros, "", data.astype(int).astype(str))
    else:
        annot = np.where(data < 1e-6, "", np.vectorize("{:.2f}".format)(data))

    if log:
        epsilon = 1e-3
        data_nonzero = data + epsilon
        norm = LogNorm(vmin=np.min(data_nonzero), vmax=np.max(data_nonzero))
    else:
        data_nonzero = data  # keep original values
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))

    # Plot
    plt.figure(figsize=(20, 12))
    ax = sns.heatmap(
        data_nonzero,
        annot=annot,
        fmt='',
        cmap='magma',
        norm=norm,
        linecolor='gray',
        cbar_kws={'label': 'value'},
        annot_kws={"size": 6},
        xticklabels=list(names),
        yticklabels=list(names)
    )

    # Configure colorbar ticks
    colorbar = ax.collections[0].colorbar
    if log:
        colorbar.locator = LogLocator(base=10)
    else:
        colorbar.locator = MaxNLocator(nbins=5)  # auto-adjusts to a reasonable number of ticks
    colorbar.update_ticks()

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("predicted")
    plt.ylabel("real")
    plt.tight_layout()
    plt.show()

def get_confusion_matrix(all_pred, all_gt, labels):
    l = all_pred.shape[0]
    res = np.zeros((l, l))
    for label in labels:
        gt = all_gt[label, :]
        hits = np.logical_and(all_pred, gt)
        counts = np.sum(hits, axis=1)
        res[label, labels] = counts[labels]
    res = res[labels, :]
    data = res[:, labels]

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=False)
    args = parser.parse_args()
    pred_file = args.file

    classes, labels, names, num_classes_orig = common.parseClassFile("/home/user/code/vlmaps/cfg/mpcat40_edit.tsv", ";", remove_aggregate_classes=True)
    labels = labels.astype(np.uint8)

    if not pred_file:
        pred_file = ""
        root = Tk()
        root.withdraw()
        #path = askdirectory(title='Select Folder', initialdir="/home/user/code/vlmaps/data/mapdata/analysis-density")
        while True:
            pred_file = askopenfilename(title='Select map', initialdir="/home/user/code/vlmaps/data/mapdata/analysis-density")
            if "all-predictions" in pred_file:
                break
            else:
                print("You must select predictions file")
        root.destroy()

    match = re.search(r"-?\b\d+\b", pred_file)
    if match:
        map_id = int(match.group())
    else:
        raise ValueError("Invalid filename - no map id")

    path = Path(pred_file).parent
    gt_file = pred_file.replace("predictions", "gts")
    grid_file = f"{path}/grid-{map_id}.npy"

    if gt_file is None or pred_file is None or grid_file is None:
        print("Could not find all files!")
        exit()

    assert gt_file is not None
    assert pred_file is not None
    assert grid_file is not None

    all_gt = np.load(f"{gt_file}")
    all_pred = np.load(f"{pred_file}")
    grid = np.load(f"{grid_file}")

    pred = all_pred[labels, :].astype(np.bool_)
    gt = all_gt[labels, :].astype(np.bool_)

    tp = np.logical_and(pred, gt)
    tn = np.logical_and(np.logical_not(gt), np.logical_not(gt))
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(np.logical_not(pred), gt)

    tps = tp.astype(np.uint8).sum(axis=0)
    tns = tn.astype(np.uint8).sum(axis=0)
    fps = fp.astype(np.uint8).sum(axis=0)
    fns = fn.astype(np.uint8).sum(axis=0)

    eps = np.ones_like(tps) * 1e-6
    accuracy = tps + tns / (tps + tns + fps + fns + eps)
    precision = tps / (tps + fps + eps)
    recall = tps / (tps + fns + eps)
    f1 = 2*tps / (2*tps + fps + fns + eps)

    confusion_matrix = None

    while True:
        print("coloring:")
        print("1: accuracy")
        print("2: precision")
        print("3: recall")
        print("4: f1")
        print("c: confusion matrix")
        print("i: Single id")
        print("n: Single name")
        print("q: quit")
        choice = input(": ")
        color = None
        if choice == "q":
            exit()
        elif choice == "1":
            color = common.color_counts(accuracy, normalize=False)
        elif choice == "2":
            color = common.color_counts(precision, normalize=False)
        elif choice == "3":
            color = common.color_counts(recall, normalize=False)
        elif choice == "4":
            color = common.color_counts(f1, normalize=False)
        elif choice == "i":
            try:
                idx = int(input("id: "))
                p = all_pred[idx, :]
                color = common.color_counts(p, normalize=True)
            except:
                print("Couldn't parse id")
                continue
        elif choice == "n":
            try:
                name = input("name: ")
                if name not in names:
                    print("Couldn't find name")
                    continue
                idx, = np.where(names == name)
                label = labels[idx.item()]
                p = all_pred[label, :]
                color = common.color_counts(p, normalize=True)
            except:
                print("Couldn't parse id")
                continue
        elif choice == "c":
            if confusion_matrix is None:
                confusion_matrix = get_confusion_matrix(all_pred, all_gt, labels)

            print("1. Linear")
            print("2. Normalized")
            print("3. Log")
            try:
                c = int(input(": "))
            except:
                c = 1

            visualize_confusion_matrix(confusion_matrix, c, names)

        else:
            print("Unknown selection")

        if color is None:
            continue

        visualize_rgb_map_3d(grid, color)



if __name__ == "__main__":
    main()