for complement in other others background texture stuff nothing object things STT_ZERO_STT
do
    ./measure_classification_density_all.sh --experiment ICRA-ablation-regular-antonyms-pe --type b --prompt_eng 1 --complement $complement --metadata $complement
    ./measure_classification_density_all.sh --experiment ICRA-ablation-regular-antonyms-nope --type b --prompt_eng 0 --complement $complement --metadata $complement
done

