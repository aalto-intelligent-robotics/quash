from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d
import scipy.misc
from scipy import ndimage
import cv2
import os
from pathlib import Path
from enum import Enum
import os


class Method(Enum):
    SIMPLE = (1,)
    SMART = 2


def createDirs(file):
    path = os.path.dirname(os.path.abspath(file))
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./semantic_image_modifier.py")
    parser.add_argument(
        "--in",
        "-i",
        dest="input",
        type=str,
        required=False,
        default=os.environ["DATA_DIR"] + "/vlmaps_dataset/5LpN3gDmAk7_1/semantic",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out",
        type=str,
        required=False,
        default=os.environ["DATA_DIR"] + "/vlmaps_dataset/5LpN3gDmAk7_1/semantic-blur",
    )
    parser.add_argument(
        "--depth",
        "-d",
        dest="depth",
        type=str,
        required=False,
        default=os.environ["DATA_DIR"] + "/vlmaps_dataset/5LpN3gDmAk7_1/depth",
    )
    parser.add_argument(
        "--force",
        "-f",
        dest="force",
        action="store_true",
        required=False,
        default=False
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input
    out = FLAGS.out
    depths = FLAGS.depth
    force = FLAGS.force

img_files = sorted([f for f in listdir(input) if isfile(join(input, f))])
depth_files = sorted([f for f in listdir(depths) if isfile(join(depths, f))])
method = Method.SIMPLE

# check if exists
if not force:
    if os.path.isdir(out):
        out_files = sorted([f for f in listdir(out) if isfile(join(out, f))])
        already_done = True
        for img in img_files:
            if img not in out_files:
                already_done = False
                break
else:
    already_done = False

if already_done:
    print("previously completed, exiting. If you want to overwrite, run with --force or -f")
    exit(0)

for idx in tqdm(range(len(img_files))):
    img = np.load(input + "/" + img_files[idx])

    orig_labels = np.unique(img)
    # print("unique values in img:", orig_labels)

    if method == Method.SIMPLE:
        img = img.astype("uint8")
        kernel = np.ones((5, 5), np.uint8)
        erd = cv2.erode(img, kernel, iterations=5)
        dil = cv2.dilate(erd, kernel, iterations=5)
        outimg = dil
    elif method == Method.SMART:
        # get depth image
        depth = np.load(depths + "/" + depth_files[idx])
        depth = depth / np.max(depth)
        # get depth edges
        edges_x = cv2.Sobel(src=depth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=9)
        edges_y = cv2.Sobel(src=depth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=9)
        edges_xy = abs(edges_x) + abs(edges_y)
        ret, edges_xy = cv2.threshold(edges_xy, 255, 1, 0)

        # use the edges to refine the label areas
        uniq_labs = np.unique(img)
        newlab = np.full_like(img, 0)
        for ulab in uniq_labs:
            ulabimg = np.where(img == ulab, 1, 0)
            hits = ulabimg * edges_xy
            corrected = ulabimg - hits
            clab = np.where(corrected > 0, True, False)
            newlab[clab] = ulab

        # apply some smoothing
        kernel = np.ones((3, 3), np.uint8)
        its = 1
        newlab = cv2.erode(newlab, kernel, iterations=its)
        newlab = cv2.dilate(newlab, kernel, iterations=its + 1)
        outimg = newlab

    outfile = out + "/" + img_files[idx]
    createDirs(outfile)
    np.save(outfile, outimg)
