import numpy as np
import matplotlib.pyplot as plt

#  ██████╗██╗      █████╗ ███████╗███████╗███████╗███████╗
# ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝
# ██║     ██║     ███████║███████╗███████╗█████╗  ███████╗
# ██║     ██║     ██╔══██║╚════██║╚════██║██╔══╝  ╚════██║
# ╚██████╗███████╗██║  ██║███████║███████║███████╗███████║
#  ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝

def parseClassFile(file, delim, remove_aggregate_classes = True):
    dont_use = [
        "void",
        "seating",
        "furniture",
        "appliances",
        "objects",
        "misc",
        "unlabeled"
    ]
    f = open(file, 'r')
    lines = f.readlines()
    first = True

    numLines = len(lines)
    labels = []
    names = []
    idx = 0
    for line in lines:
        if (first):
            first = False
            continue

        parts = line.split(delim)
        label = int(parts[0])
        name = parts[1].rstrip("\n").lstrip(" ").rstrip(" ")
        if remove_aggregate_classes and name in dont_use:
            continue
        labels.append(label)
        names.append(name)
        idx += 1
    classes = np.array((labels, names))
    f.close()

    indices = np.argsort(classes[0, :].astype(np.int32))
    classes = classes[:, indices]

    labels = classes[0, :]
    names = classes[1, :]

    if not remove_aggregate_classes:
        names = names[1:-1]
        labels = labels[1:-1]

    num_classes_orig = len(lines)-1

    return classes, labels, names, num_classes_orig

#  ██████╗ ██████╗ ██╗      ██████╗ ██████╗ ███████╗
# ██╔════╝██╔═══██╗██║     ██╔═══██╗██╔══██╗██╔════╝
# ██║     ██║   ██║██║     ██║   ██║██████╔╝███████╗
# ██║     ██║   ██║██║     ██║   ██║██╔══██╗╚════██║
# ╚██████╗╚██████╔╝███████╗╚██████╔╝██║  ██║███████║
#  ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝

def color_semantics(grid, onedim = False):
    sem_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        if(onedim):
            label = grid[i]
        else:
            label = grid[i, 0]
        sem_cols[i, :] = color_semantics_single(label)
    return sem_cols

def color_semantics_single(label):
    try:
        color_tuple = colormap[label]
        color = (color_tuple["r"], color_tuple["g"], color_tuple["b"])
    except KeyError:
        color = (0, 0, 0)
    return color

def color_regions(grid):
    reg_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        label = grid[i, 0]
        try:
            col = colormap[label]
            reg_cols[i, :] = (col["r"], col["g"], col["b"])
        except KeyError:
            reg_cols[i, :] = (0, 0, 0)
    return reg_cols


def color_instances(grid, onedim = False):
    # print(grid.shape)
    insta_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        if(onedim):
            instance = grid[i]
        else:
            instance = grid[i, 0]

        insta_cols[i, :] = color_instances_single(instance)

        # im = instance % 255
        # insta_cols[i, :] = (im, im, im)
    return insta_cols

def color_instances_single(instance):
    idx = instance % 41
    color_tuple = colormap[idx]
    color = (color_tuple["r"], color_tuple["g"], color_tuple["b"])
    return color

def color_contrastive(gt, pred):
    sem_cols = np.zeros((gt.shape[0], 3))
    idx = gt == pred
    diff = np.full_like(gt, False)
    diff[idx] = True
    for i in range(diff.shape[0]):
        if(diff[i]):
            sem_cols[i, :] = (0, 255, 0)
        else:
            sem_cols[i, :] = (255, 0, 0)
    return sem_cols


def color_counts(values, normalize=False, log=False, log_exp=1.0):
    values = np.asarray(values)

    if normalize:
        div = (values.max() - values.min())
        if div == 0:
            div = 1e-6
        values = (values - values.min()) / div

    if log:
        values = np.log1p(values) ** log_exp

    cmap = plt.get_cmap('viridis')
    rgb_array = cmap(values)[:, :3] * 255
    return rgb_array.astype(np.uint8)


colormap = {}
# colormap[0] =  {"r": 0, "g": 0, "b": 0}
# colormap[1] =  {"r": 255, "g": 0, "b": 0}
# colormap[2] =  {"r": 0, "g": 255, "b": 0}
# colormap[3] =  {"r": 0, "g": 0, "b": 255}
# colormap[4] =  {"r": 255, "g": 255, "b": 0}
# colormap[5] =  {"r": 255, "g": 0, "b": 255}
# colormap[6] =  {"r": 0, "g": 255, "b": 255}
# colormap[7] =  {"r": 255, "g": 255, "b": 255}
# colormap[8] =  {"r": 200, "g": 0, "b": 0}
# colormap[9] =  {"r": 0, "g": 200, "b": 0}
# colormap[10] = {"r": 0, "g": 0, "b": 200}
# colormap[11] = {"r": 200, "g": 200, "b": 0}
# colormap[12] = {"r": 200, "g": 0, "b": 200}
# colormap[13] = {"r": 0, "g": 200, "b": 200}
# colormap[14] = {"r": 200, "g": 200, "b": 200}
# colormap[15] = {"r": 150, "g": 0, "b": 0}
# colormap[16] = {"r": 0, "g": 150, "b": 0}
# colormap[17] = {"r": 0, "g": 0, "b": 150}
# colormap[18] = {"r": 150, "g": 150, "b": 0}
# colormap[19] = {"r": 150, "g": 0, "b": 150}
# colormap[20] = {"r": 0, "g": 150, "b": 150}
# colormap[21] = {"r": 150, "g": 150, "b": 150}
# colormap[22] = {"r": 100, "g": 0, "b": 0}
# colormap[23] = {"r": 0, "g": 100, "b": 0}
# colormap[24] = {"r": 0, "g": 0, "b": 100}
# colormap[25] = {"r": 100, "g": 100, "b": 0}
# colormap[26] = {"r": 100, "g": 0, "b": 100}
# colormap[27] = {"r": 0, "g": 100, "b": 100}
# colormap[28] = {"r": 100, "g": 100, "b": 100}
# colormap[29] = {"r": 50, "g": 0, "b": 0}
# colormap[30] = {"r": 0, "g": 50, "b": 0}
# colormap[31] = {"r": 0, "g": 0, "b": 50}
# colormap[32] = {"r": 50, "g": 50, "b": 0}
# colormap[33] = {"r": 50, "g": 0, "b": 50}
# colormap[34] = {"r": 0, "g": 50, "b": 50}
# colormap[35] = {"r": 50, "g": 50, "b": 50}
# colormap[36] = {"r": 25, "g": 0, "b": 0}
# colormap[37] = {"r": 0, "g": 25, "b": 0}
# colormap[38] = {"r": 0, "g": 0, "b": 25}
# colormap[39] = {"r": 25, "g": 25, "b": 0}
# colormap[40] = {"r": 25, "g": 0, "b": 25}
# colormap[41] = {"r": 0, "g": 25, "b": 25}



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
