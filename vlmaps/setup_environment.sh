#!/bin/bash

if [ -z $1 ]
then
    export BASE_DIR="/home/user"
else
    export BASE_DIR="#FIXME"
fi

export VLMAPS_DIR="$BASE_DIR/code/vlmaps"
export DATA_DIR="$BASE_DIR/hdd/datasets"
export OPENSCENE_DIR="$BASE_DIR/code/openscene"
export TF_CPP_MIN_LOG_LEVEL=3

source $BASE_DIR/<path/to/quash>/setup_env.sh
source $BASE_DIR/<path/to/quash>/openai_key.sh