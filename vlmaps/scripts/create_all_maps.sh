#!/bin/bash

for map in 0 1 2 3 4 5 6 7 8 9
do
    for encoder in vlmaps_lseg vlmaps_openseg
    do
        for voxel_size in 0.05 #0.02 0.05 0.1 0.2
        do
            ./create_map.sh $map $encoder $voxel_size
        done
    done
done