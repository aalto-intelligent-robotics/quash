from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import os

infile = os.environ['VLMAPS_DIR'] + "/data/mpcat40_edit.tsv"
outfile = open("../data/labels.cfg", "w")

colormap = True

data = open(infile)
first = True
for i in data:
    if(first):
        first = False
        continue

    x = i.split(";")
    print(x)

    if(colormap):
        hex = x[2].replace("#", "")
        strr = hex[0:2]
        strg = hex[2:4]
        strb = hex[4:6]
        print(strr, strg, strb)

        r = int(strr, 16)
        g = int(strg, 16)
        b = int(strb, 16)
        print(r, g, b)
        outfile.write("{" + x[0] + ", {" + str(r) + ", " + str(g) + ", "+ str(b) + "}},")
        outfile.write("\n")
    else:
        outfile.write(x[0] + ", ")
