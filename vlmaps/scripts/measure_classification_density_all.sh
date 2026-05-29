#!/bin/bash
source ./argparse.sh
define_arg "experiment" "" "Experiment name" "string" "optional"
define_arg "classifier_config" "" "Classifier config" "string" "optional"
define_arg "type" "comp" "Comparison type: comp/method/baseline/method-var/baseline-var/all" "string" "optional"
define_arg "classifier" "cosine-svm" "Classifier" "string" "optional"
define_arg "prompt_eng" "0" "Prompt engineering type" "int" "optional"
define_arg "flat" "false" "Use flat dataset" "bool" "optional"
define_arg "prompts" "default" "Used prompts" "string" "optional"
define_arg "synonym" "0" "Synonym type" "int" "optional"
define_arg "antonym" "0" "Antonym type" "int" "optional"
define_arg "transformer" "" "Transformer type" "str" "optional"
define_arg "sample" "" "use sample map" "bool" "optional"
define_arg "median" "" "use median map" "bool" "optional"
define_arg "complement" "" "complement, csv" "str" "optional"
define_arg "metadata" "" "metadata" "str" "optional"
define_arg "postprocessing" "false" "Use postprocessing" "bool" "optional"
define_arg "twod" "false" "Index in 2d" "bool" "optional"
define_arg "encoder" "both" "Used encoders" "str" "optional"

# [Optional] Check for -h and --help
check_for_help "$@"

# Parse the arguments
parse_args "$@"

if [ -z $experiment ]
then
    exp=
else
    exp="--experiment $experiment"
fi

if [ -z $classifier_config ]
then
    cc=
else
    cc="--classifier_config $classifier_config"
fi

if [[ $flat = "true" ]]
then
    flat_setting="--flat"
else
    flat_setting=""
fi

if [ -z $transformer ]
then
    tf=
else
    tf="--transformer $transformer"
fi

if [ -z $sample ]
then
    sample_map=
else
    sample_map="--sample"
fi

if [ -z $median ]
then
    median_map=
else
    median_map="--median"
fi

if [ -z $complement ]
then
    complement_str=
else
    complement_str="--complement $complement"
fi

if [ -z $metadata ]
then
    metadata_str=
else
    metadata_str="--metadata $metadata"
fi

if [[ -z "$encoder" || "$encoder" == "both" ]]
then
    encoders=("vlmaps_lseg" "vlmaps_openseg")
else
    encoders=("$encoder")
fi

# mean prompt engineering
default_prompt_type=1

for map in 0 1 2 3 4 5 6 7 8 9
do
    for run_encoder in ${encoders[@]}
    do
        for voxel_size in 0.05 #0.02 0.05 0.1 0.2
        do
            if [ $type == "comp" ] || [ $type == "c" ]
            then
                # main comparison
                #####################################
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --density --classifier $classifier $cc --prompt_eng $prompt_eng --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --prompt_eng 1
            elif [ $type == "method" ] || [ $type == "m" ]
            then
                # only method - best
                #####################################
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --prompt_eng $prompt_eng --synonym $synonym --antonym $antonym
            elif [ $type == "baseline" ] || [ $type == "b" ]
            then
                # only baseline - best
                #####################################
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --prompt_eng $prompt_eng
                #./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --prompt_eng 1
            elif [ $type == "method-var" ] || [ $type == "mv" ]
            then
                # all method combinations
                #####################################
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --postprocessing --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --postprocessing --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --postprocessing --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --postprocessing --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
            elif [ $type == "baseline-var" ] || [ $type == "bv" ]
            then
                # all BL combinations
                #####################################
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --postprocessing
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --prompt_eng 1
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod --postprocessing
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod --prompt_eng 1
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --postprocessing --prompt_eng 1
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod --postprocessing --prompt_eng 1
            elif [ $type == "all" ] || [ $type == "a" ]
            then
                # all combinations
                #####################################
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --postprocessing --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --postprocessing --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --postprocessing --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts $cc --density --classifier $classifier --twod --postprocessing --prompt_eng $default_prompt_type --synonym $synonym --antonym $antonym
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --postprocessing
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --prompt_eng $default_prompt_type
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod --postprocessing
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod --prompt_eng $default_prompt_type
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --postprocessing --prompt_eng $default_prompt_type
                ./measure_classification.sh --id $map --encoder $run_encoder --voxel_size $voxel_size $exp $sample_map $median_map $complement_str $metadata_str $flat_setting $tf --prompts $prompts --twod --postprocessing --prompt_eng $default_prompt_type
            else
                echo "Unknown type [comp/mv/all]"
            fi
        done
    done
done