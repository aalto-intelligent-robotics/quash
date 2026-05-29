from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from open3d import *

ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
