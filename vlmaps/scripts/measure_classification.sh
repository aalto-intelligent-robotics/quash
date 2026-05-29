#!/bin/bash
source ./argparse.sh

# Define an argument
#define_arg "name" "default" "description" "type" "optional/required"
define_arg "id" "0" "Map id" "int" "optional"
define_arg "encoder" "vlmaps_lseg" "Encoder" "string" "optional"
define_arg "voxel_size" "0.05" "Voxel size" "string" "optional"
define_arg "force" "false" "Force overwrite" "bool" "optional"
define_arg "prompt_eng" "0" "Prompt engineering type" "int" "optional"
define_arg "postprocessing" "false" "Use postprocessing" "bool" "optional"
define_arg "twod" "false" "Index in 2d" "bool" "optional"
define_arg "density" "false" "Use our method" "bool" "optional"
define_arg "experiment" "" "Experiment name" "string" "optional"
define_arg "classifier_config" "" "Classifier config" "string" "optional"
define_arg "classifier" "cosine-svm" "Classifier" "string" "optional"
define_arg "flat" "false" "Use flat dataset" "bool" "optional"
define_arg "prompts" "default" "Used prompts" "string" "optional"
define_arg "synonym" "0" "Synonym type" "int" "optional"
define_arg "antonym" "0" "Antonym type" "int" "optional"
define_arg "transformer" "" "Transformer type" "str" "optional"
define_arg "sample" "" "use sample map" "bool" "optional"
define_arg "median" "" "use median map" "bool" "optional"
define_arg "complement" "" "complement, csv" "str" "optional"
define_arg "metadata" "" "metadata" "str" "optional"

# [Optional] Check for -h and --help
check_for_help "$@"

# Parse the arguments
parse_args "$@"

gs=$(echo 125/$voxel_size|bc)

if [ -z $experiment ]
then
    exp=
else
    exp="experiment=$experiment"
fi

if [ -z $transformer ]
then
    tf=
else
    tf="transformer=$transformer"
fi

if [[ $flat = "true" ]]
then
    config="--config-name measure_classification_flat.yaml"
else
    config="--config-name measure_classification.yaml"
fi

if [ -z $sample ]
then
    sample_map=
else
    sample_map="sample=True"
fi

if [ -z $median ]
then
    median_map=
else
    median_map="median=True"
fi

if [ -z $complement ]
then
    complement_str=
else
    complement_str="complement=$complement"
fi

if [ -z $metadata ]
then
    metadata_str=
else
    metadata_str="metadata=$metadata"
fi

cd ../vlmaps

echo ""
echo ""
echo ""
echo ""
echo ""
echo "*********************************************************************************"
echo python -m application.measure_classification \
          $config \
          "scene_id=$id" \
          "map_config/visual_encoder=$encoder" \
          "force=$force" \
          "params.cs=$voxel_size" \
          "params.gs=$gs" \
          "prompt_engineering=$prompt_eng" \
          "use_postprocessing=$postprocessing" \
          "index_2d=$twod" \
          "density=$density" \
          "classifier_config=$classifier_config" \
          "classifier=$classifier" \
          "prompts=$prompts" \
          "synonym_type=$synonym" \
          "antonym_type=$antonym" \
          $exp \
          $tf \
          $sample_map \
          $median_map \
          $complement_str \
          $metadata_str
echo "*********************************************************************************"

python -m application.measure_classification \
          $config \
          "scene_id=$id" \
          "map_config/visual_encoder=$encoder" \
          "force=$force" \
          "params.cs=$voxel_size" \
          "params.gs=$gs" \
          "prompt_engineering=$prompt_eng" \
          "use_postprocessing=$postprocessing" \
          "index_2d=$twod" \
          "density=$density" \
          "classifier_config=$classifier_config" \
          "classifier=$classifier" \
          "prompts=$prompts" \
          "synonym_type=$synonym" \
          "antonym_type=$antonym" \
          $exp \
          $tf \
          $sample_map \
          $median_map \
          $complement_str \
          $metadata_str