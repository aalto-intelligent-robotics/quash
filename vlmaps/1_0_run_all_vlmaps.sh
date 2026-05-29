#!/bin/bash
# data directory
if [ -z $1 ]
then
    echo "Usage: ./run_all <data_dir> <voxel_size> <force> <nodl>"
    exit 1
else
    dir=$1
fi

# Voxel size
if [ -z $2 ]
then
    voxel_size=0.05
else
    voxel_size=$2
fi

force=0
if ! [ -z $3 ] && [ $3 == "force" ]
then
    force=1
fi

nodownload=0
if ! [ -z $4 ] && [ $4 == "nodl" ]
then
    nodownload=1
fi

forcedownload=0
if [ $nodownload -eq 1 ]
then
    forcedownload=0
else
    if [ $force -eq 1 ]
    then
        forcedownload=1
    fi
fi

##################################################
# DATA: Downloading, Generation and Preprocessing
##################################################

echo "================================="
echo "Running prerequisites"
echo "================================="

cd scripts

echo "1/4 Download dataset"
echo "---------------------------"
# Download all Matterport scenes processed by VLMaps
if [ -e $dir/matterport3d/v1/ ] && [ $forcedownload -eq 0 ]
then
    echo "Data folder exists. Skip downloading."
else
    echo "Downloading data..."
    ./downloader.sh $dir/matterport3d/
fi

echo " "
echo "2/4 Unzip"
echo "---------------------------"
# Unzip data
if [ -e $dir/matterport3d/v1/tasks/mp3d/ ] && [ $forcedownload -eq 0 ]
then
    echo "mp3d exists."
else
    echo "Unzipping mp3d..."
    unzip $dir/matterport3d/v1/tasks/mp3d_habitat.zip -d $dir/matterport3d/v1/tasks/
fi

echo " "
echo "3/4 Check dataset"
echo "---------------------------"
# Check dataset
./check_initial_dataset.sh
success=$?

echo " "
echo "4/4 Generate dataset"
echo "---------------------------"
if [ ${success} -ne 0 ] && [ $force -eq 0 ]
then
    # Generate dataset
    ./generate_dataset.sh
else
    echo "done"
fi


##################################################
# MAPS: Map creation, Postprocessing, Analysis
##################################################

for encoder in vlmaps_lseg vlmaps_openseg
do
    echo " "
    echo "================================="
    echo "Encoder: $encoder"
    echo "================================="

    for i in 0 1 2 3 4 5 6 7 8 9
    do
        echo " "
        echo "================================="
        echo "Map: $((i+1))/10"
        echo "================================="
        ./run_one_scene.sh $i $encoder $voxel_size $3
    done
done
