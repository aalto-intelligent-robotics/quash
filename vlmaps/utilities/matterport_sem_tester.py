from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from open3d import *
import matplotlib.image as mpimg
import os

colormap = {}
colormap[0] = {"r": 255, "g": 255, "b": 255}
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
# create histograms from ndt maps

def display3d(img, label):
    fx_inv = 1 / 540
    fy_inv = 1 / 540
    cx = 540
    cy = 360

    print(fx_inv)
    print(fy_inv)

    h = img.shape[0]
    w = img.shape[1]
    cloud = np.full((h*w, 3),-1,dtype=np.float32)
    colors = np.full((h*w, 3), -1, dtype=np.float32)

    idx = 0
    for i in tqdm(range(h)):
        for j in range(w):
            #points
            z = img[i,j]
            x = (j - cx) * z * fx_inv
            y = (i - cy) * z * fy_inv
            cloud[idx, 0] = x
            cloud[idx, 1] = y
            cloud[idx, 2] = z

            #coloring
            l = label[i, j]
            if(l < 0):
                l = 0
            cm = colormap[l]
            colors[idx, 0] = cm["r"]/255
            colors[idx, 1] = cm["g"]/255
            colors[idx, 2] = cm["b"]/255

            idx += 1

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([pcd])

def distance(a, b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./matterport_conversion.py")
    parser.add_argument(
        '--images', '-i',
        dest="images",
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
    parser.add_argument(
        '--rgbs', '-r',
        dest="rgbs",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/rgb"
    )
    parser.add_argument(
        '--regions', '-rg',
        dest="regions",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/regions"
    )
    parser.add_argument(
        '--embeddings', '-e',
        dest="embeddings",
        type=str,
        required=False,
        default= os.environ['DATA_DIR'] + "/vlmaps/5Lp_embeddings"
    )
    parser.add_argument(
        '--mode', '-m',
        dest="mode",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '--idx', '-d',
        dest="idx",
        type=int,
        required=False,
        default=0
    )
    FLAGS, unparsed = parser.parse_known_args()
    images = FLAGS.images
    labels = FLAGS.labels
    embeddings = FLAGS.embeddings
    mode = FLAGS.mode
    rgbs = FLAGS.rgbs
    regions = FLAGS.regions
    idx = FLAGS.idx

img_files = sorted([f for f in listdir(images) if isfile(join(images, f))])
label_files = sorted([f for f in listdir(labels) if isfile(join(labels, f))])
rgb_files = sorted([f for f in listdir(rgbs) if isfile(join(rgbs, f))])
region_files = sorted([f for f in listdir(regions) if isfile(join(regions, f))])

if(mode == 1):
    print("loading embedding...")
    embedding_files = sorted([f for f in listdir(embeddings) if isfile(join(embeddings, f))])

inp = ''
while (inp != 'q'):
    img = np.load(images+"/"+img_files[idx])
    label = np.load(labels+"/"+label_files[idx])
    rgb = mpimg.imread(rgbs+"/"+rgb_files[idx])
    reg = np.load(regions+"/"+region_files[idx])

    if(mode == 1):
        embedding = np.fromfile(embeddings+"/"+embedding_files[idx], dtype=np.float32)
        embedding = embedding.reshape(720,1080,512)
        distances = np.full((720,1080),-1,dtype=np.float32)
        comp = embedding[0,0,:]
        #print(comp)
        printed = True
        for r in tqdm(range(720)):
            for c in range(1080):
                e = embedding[r,c,:]
                d = distance(e, comp)
                distances[r,c] = d
                if(d != 1 and not printed):
                    print("e",e)
                    print("d",d)
                    printed=True

        print(np.min(distances))
        print(np.max(distances))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(img)
        ax2.imshow(label)
        ax3.hist(distances.reshape(-1), bins=100)
        ax4.imshow(distances)
        plt.show()

    if(mode == 0 or mode == 2):
        print(np.min(label))
        print(np.max(label))
        print(img.shape)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        ax1.imshow(img)
        ax2.imshow(label)
        ax3.imshow(rgb)
        ax4.imshow(reg)
        plt.show()

        if(mode == 2):
            display3d(img, label)

    inp = input(": ")
    if(inp == 'a'):
        idx -= 1
    elif (inp == 'd'):
        idx += 1
    if(idx < 0):
        idx == 0
    if(idx > len(img_files)):
        idx = len(img_files)
exit()

