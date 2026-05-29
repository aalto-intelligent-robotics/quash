from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import os

# create histograms from ndt maps

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./matterport_conversion.py")
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/matterport3d/processed/5LpN3gDmAk7_1/depth"
    )
    parser.add_argument(
        '--output', '-o',
        dest="output",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/matterport3d/processed/5LpN3gDmAk7_1/depth_bin"
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input
    output = FLAGS.output

files = [f for f in listdir(input) if isfile(join(input, f))]

if not exists(output):
    makedirs(output)

for file in tqdm(files):
    infile = input+"/"+file
    outfile = output+"/"+file.replace("npy", "bin")
    arr = np.load(infile)
    u = np.unique(arr)
    print(u)
    print(arr.dtype)
    print(arr.shape)
    exit
    #arr.tofile(outfile)
