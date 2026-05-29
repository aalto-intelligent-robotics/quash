from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import Normalize
import cv2
import os

def normalize(M):
    N = (M - np.min(M)) / (np.max(M) - np.min(M))
    return N

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./sem_depth.py")
    parser.add_argument(
        '--images', '-i',
        dest="images",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/rgb"
    )
    parser.add_argument(
        '--depth', '-d',
        dest="depth",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/depth"
    )
    parser.add_argument(
        '--labels', '-l',
        dest="labels",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/semantic"
    )
    FLAGS, unparsed = parser.parse_known_args()
    images = FLAGS.images
    labels = FLAGS.labels
    depths= FLAGS.depth

img_files = sorted([f for f in listdir(images) if isfile(join(images, f))])
label_files = sorted([f for f in listdir(labels) if isfile(join(labels, f))])
depth_files = sorted([f for f in listdir(depths) if isfile(join(depths, f))])

idx = 40
inp = ''
while (inp != 'q'):
    #! LOAD IMAGES
    img = mpimg.imread(images+"/"+img_files[idx])
    label = np.load(labels+"/"+label_files[idx])
    nlabel = normalize(label)
    clabel = cm.viridis(nlabel)

    depth = np.load(depths+"/"+depth_files[idx])
    #depth = depth.astype(dtype=np.uint32)
    depth = depth / np.max(depth)
    # print(depth.dtype, np.min(depth), np.max(depth))

    #! EDGES
    #edges_xy = cv2.Sobel(src=depth, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=9)
    edges_x = cv2.Sobel(src=depth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=9)
    edges_y = cv2.Sobel(src=depth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=9)
    edges_xy = abs(edges_x) + abs(edges_y)
    # print(edges_xy.dtype, np.min(edges_xy), np.max(edges_xy))
    # print(edges_x.dtype, np.min(edges_x), np.max(edges_x))
    # print(edges_y.dtype, np.min(edges_y), np.max(edges_y))
    ret, edges_xy = cv2.threshold(edges_xy, 255, 1, 0)
    # kernel = np.ones((2, 2), np.uint8)
    # edges_xy = cv2.erode(edges_xy, kernel, iterations=3)
    # ret, thresh = cv2.threshold(depth, 127, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)

    uniq_labs = np.unique(label)
    newlab = np.full_like(label, 0)
    for ulab in uniq_labs:
        ulabimg = np.where(label == ulab, 1, 0)
        print(ulabimg.dtype, np.min(ulabimg), np.max(ulabimg))
        hits = ulabimg * edges_xy
        print(hits.dtype, np.min(hits), np.max(hits))
        corrected = ulabimg - hits
        print(corrected.dtype, np.min(corrected), np.max(corrected))
        clab = np.where(corrected > 0, True, False)
        newlab[clab] = ulab

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # ax1.imshow(ulabimg)
        # ax2.imshow(edges_xy)
        # ax3.imshow(hits)
        # ax4.imshow(corrected)
        # plt.show()

    kernel = np.ones((3, 3), np.uint8)
    its = 1
    newlab = cv2.erode(newlab, kernel, iterations=its)
    newlab = cv2.dilate(newlab, kernel, iterations=its+1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(label)
    ax2.imshow(newlab)
    plt.show()

    #! PLOT
    # normalization = "linear"
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.imshow(depth)
    # ax2.imshow(edges_xy, norm=normalization)
    # ax3.imshow(label)
    # ax4.imshow(label+edges_xy)
    # plt.show()
    # try:
    #     cv2.imshow("depth", depth)
    #     cv2.imshow("edges", edges)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # except:
    #     print("failed")

    inp = input(": ")
    if (inp == 'a'):
        idx -= 1
    elif (inp == 'd'):
        idx += 1
    elif (inp.isdigit()):
        idx = int(inp)
    if (idx < 0):
        idx == 0
    if (idx > len(img_files)):
        idx = len(img_files)
exit()
