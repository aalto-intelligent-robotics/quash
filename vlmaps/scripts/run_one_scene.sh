#!/bin/bash
# scene_id
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

# Visual encoder config
if [ -z $2 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$2
fi

# Voxel size
if [ -z $3 ]
then
    voxel_size=0.05
else
    voxel_size=$3
fi

force=$4

# Create the base map
########################################################################################
echo " "
echo "step 1/1: create map"
echo "---------------------------"
./create_map.sh $id $encoder $voxel_size $force