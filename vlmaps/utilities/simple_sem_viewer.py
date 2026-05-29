from os import listdir
from os.path import isfile, join
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import Normalize
from pynput import keyboard
import os

colormap = {}
colormap[0] = {"r": 120, "g": 120, "b": 120}
colormap[1] = {"r": 174, "g": 199, "b": 232}
colormap[2] = {"r": 112, "g": 128, "b": 144}
colormap[3] = {"r": 152, "g": 223, "b": 138}
colormap[4] = {"r": 197, "g": 176, "b": 213}
colormap[5] = {"r": 255, "g": 127, "b": 14}
colormap[6] = {"r": 214, "g": 39, "b": 40}
colormap[7] = {"r": 31, "g": 119, "b": 180}
colormap[8] = {"r": 188, "g": 189, "b": 34}
colormap[9] = {"r": 255, "g": 152, "b": 150}
colormap[10] = {"r": 44, "g": 160, "b": 44}
colormap[11] = {"r": 227, "g": 119, "b": 194}
colormap[12] = {"r": 222, "g": 158, "b": 214}
colormap[13] = {"r": 148, "g": 103, "b": 189}
colormap[14] = {"r": 140, "g": 162, "b": 82}
colormap[15] = {"r": 132, "g": 60, "b": 57}
colormap[16] = {"r": 158, "g": 218, "b": 229}
colormap[17] = {"r": 156, "g": 158, "b": 222}
colormap[18] = {"r": 231, "g": 150, "b": 156}
colormap[19] = {"r": 99, "g": 121, "b": 57}
colormap[20] = {"r": 140, "g": 86, "b": 75}
colormap[21] = {"r": 219, "g": 219, "b": 141}
colormap[22] = {"r": 214, "g": 97, "b": 107}
colormap[23] = {"r": 206, "g": 219, "b": 156}
colormap[24] = {"r": 231, "g": 186, "b": 82}
colormap[25] = {"r": 57, "g": 59, "b": 121}
colormap[26] = {"r": 165, "g": 81, "b": 148}
colormap[27] = {"r": 173, "g": 73, "b": 74}
colormap[28] = {"r": 181, "g": 207, "b": 107}
colormap[29] = {"r": 82, "g": 84, "b": 163}
colormap[30] = {"r": 189, "g": 158, "b": 57}
colormap[31] = {"r": 196, "g": 156, "b": 148}
colormap[32] = {"r": 247, "g": 182, "b": 210}
colormap[33] = {"r": 107, "g": 110, "b": 207}
colormap[34] = {"r": 255, "g": 187, "b": 120}
colormap[35] = {"r": 199, "g": 199, "b": 199}
colormap[36] = {"r": 140, "g": 109, "b": 49}
colormap[37] = {"r": 231, "g": 203, "b": 148}
colormap[38] = {"r": 206, "g": 109, "b": 189}
colormap[39] = {"r": 23, "g": 190, "b": 207}
colormap[40] = {"r": 127, "g": 127, "b": 127}
colormap[41] = {"r": 0, "g": 0, "b": 0}

def get_colormap():
    # Create a (42, 3) array of normalized RGB values
    max_index = max(colormap.keys()) + 1
    color_array = np.zeros((max_index, 3))

    for i in range(max_index):
        rgb = colormap[i]
        color_array[i] = [rgb["r"], rgb["g"], rgb["b"]]

    # Normalize to [0, 1]
    color_array = color_array / 255.0
    return color_array

def normalize(M):
    N = (M - np.min(M)) / (np.max(M) - np.min(M))
    return N

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR", default="/home/user/hdd/datasets/")
    parser = argparse.ArgumentParser("./matterport_conversion.py")
    parser.add_argument(
        "--images",
        "-i",
        dest="images",
        type=str,
        required=False,
        default=data_dir + "/vlmaps_dataset/5LpN3gDmAk7_1/rgb",
    )
    parser.add_argument(
        "--depth",
        "-d",
        dest="depth",
        type=str,
        required=False,
        default=data_dir + "/vlmaps_dataset/5LpN3gDmAk7_1/depth",
    )
    parser.add_argument(
        "--labels",
        "-l",
        dest="labels",
        type=str,
        required=False,
        default=data_dir + "/vlmaps_dataset/5LpN3gDmAk7_1/semantic",
    )
    parser.add_argument(
        "--folder",
        "-f",
        dest="folder",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--raytrace",
        "-r",
        dest="raytrace",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--compare",
        "-c",
        dest="compare",
        action="store_true",
        required=False,
        default=False
    )
    FLAGS, unparsed = parser.parse_known_args()
    images = FLAGS.images
    labels = FLAGS.labels
    depths = FLAGS.depth
    compare = FLAGS.compare
    folder = FLAGS.folder
    raytrace = FLAGS.raytrace

    if folder:
        images = f"{folder}/rgb"
        labels = f"{folder}/semantic"
        depths = f"{folder}/depth"

    if raytrace:
        complement = ""
        images += "-rt"
        labels += "-rt"
        depths += "-rt"
    else:
        complement = "-rt"

    img_files = sorted([f for f in listdir(images) if isfile(join(images, f))])
    label_files = sorted([f for f in listdir(labels) if isfile(join(labels, f))])
    depth_files = sorted([f for f in listdir(depths) if isfile(join(depths, f))])

    idx = 0
    inp = ""
    while inp != "q":
        img = mpimg.imread(images + "/" + img_files[idx])
        label = np.load(labels + "/" + label_files[idx]).squeeze()
        depth = np.load(depths + "/" + depth_files[idx])

        if np.all(depth == 0):
            print("All zero depth")

        depth_mask = depth > 100
        if np.any(depth_mask):
            depth[depth_mask] = 0
            print(f"{depth_mask.sum()} pixels fixed")

        rows = 2 if compare else 1

        plt.ion()
        fig, ax = plt.subplots(rows, 3, num=0)
        if not compare:
            ax = ax.reshape(1, -1)
        ax[0, 0].imshow(img)
        ax[0, 1].imshow(get_colormap()[label])
        #ax[0, 1].imshow(label)
        ax[0, 2].imshow(depth)

        print(np.unique(label))

        if compare:
            img_rt = mpimg.imread(images + complement + "/" + img_files[idx])
            label_rt = np.load(labels + complement + "/" + label_files[idx]).squeeze()
            depth_rt = np.load(depths + complement + "/" + depth_files[idx])
            ax[1, 0].imshow(img_rt)
            ax[1, 1].imshow(get_colormap()[label])
            #ax[0, 1].imshow(label)
            ax[1, 2].imshow(depth_rt)

        fig.canvas.draw()
        fig.canvas.flush_events()

        with keyboard.Events() as events:
            # Block for as much as possible
            event = events.get(1e6)
            assert event
            if event.key == keyboard.KeyCode.from_char('a'):
                idx -= 1
            elif event.key == keyboard.KeyCode.from_char('d'):
                idx += 1
            elif event.key == keyboard.KeyCode.from_char('j'):
                inp = input("Jump to:")
                if inp.isdigit():
                    idx = int(inp)
            elif event.key == keyboard.KeyCode.from_char('q'):
                inp = "q"
                exit()
            if idx < 0:
                idx == 0
            if idx > len(img_files):
                idx = len(img_files)

