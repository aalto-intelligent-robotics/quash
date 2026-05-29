#!/bin/bash
source ./argparse.sh
define_arg "experiment" "" "Experiment name" "string" "optional"
define_arg "classifier_config" "" "Classifier config" "string" "optional"
define_arg "type" "comp" "Comparison type: comp/method/baseline/method-var/baseline-var/all" "string" "optional"
define_arg "classifier" "cosine-svm" "Classifier" "string" "optional"
define_arg "prompt_eng" "0" "Prompt engineering type" "int" "optional"
define_arg "prompts" "default" "Used prompts" "string" "optional"

# [Optional] Check for -h and --help
check_for_help "$@"

# Parse the arguments
parse_args "$@"

if [ -z $classifier_config ]
then
    cc=
else
    cc="--classifier_config $classifier_config"
fi

for s in 0 1 2 3
do
    for a in 0 1 2 3
    do
        ./measure_classification_density_all.sh --experiment ICRA-ablation-synonyms-$s-$a $cc --type $type --classifier $classifier --prompt_eng $prompt_eng --prompts $prompts --synonym $s --antonym $a
    done
done