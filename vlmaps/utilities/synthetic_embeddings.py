from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os

# Short script to create synthetic embeddings for testing

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./synthetic_embeddings.py")
    parser.add_argument(
        '--ones', '-o',
        dest="ones",
        action='store_true',
        required=False,
    )
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=False,
        default= os.environ['DATA_DIR'] + "/vlmaps/coffee_embeddings"
    )
    parser.add_argument(
        '--output', '-o',
        dest="output",
        type=str,
        required=False,
        default= os.environ['DATA_DIR'] + "/vlmaps/coffee_embeddings_synthetic_ones/"
    )
    FLAGS, unparsed = parser.parse_known_args()
    ones = FLAGS.ones
    input = FLAGS.input
    output = FLAGS.output

print("reading...")
files = [f for f in listdir(input) if isfile(join(input, f))]
print("done!")

if(ones):
    arr = np.ones((480, 640, 512), dtype=np.float32)
else:
    arr = np.ones((480, 640, 512), dtype=np.float32)

basepath = output + "ones.bin"
arr.tofile(basepath)
print("wrote to", basepath)

for file in tqdm(files):
    fp = output + file
    Path(fp).symlink_to(basepath)

