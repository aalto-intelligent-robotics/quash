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
gs=$(echo 125/$voxel_size|bc)

force="False"
if ! [ -z $4 ] && [ $4 == "force" ]
then
    force="True"
fi

cd ../vlmaps
python -m application.create_map "scene_id=$id" "map_config/visual_encoder=$encoder" "force=$force" "params.cs=$voxel_size" "params.gs=$gs"