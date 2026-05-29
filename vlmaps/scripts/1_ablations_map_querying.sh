./measure_classification_density_all.sh --experiment ICRA-ablation-regular-queryability --type baseline --prompt_eng 0 --metadata baseline
./measure_classification_density_all.sh --experiment ICRA-ablation-regular-queryability --type method --prompt_eng 0 --synonym 0 --antonym 0 --metadata LLM
./measure_classification_density_all.sh --experiment ICRA-ablation-regular-queryability --type method --prompt_eng 0 --synonym 1 --antonym 1 --metadata gt
./measure_classification_density_all.sh --experiment ICRA-ablation-regular-queryability --type baseline --prompt_eng 1 --metadata baseline-pe
./measure_classification_density_all.sh --experiment ICRA-ablation-regular-queryability --type method --prompt_eng 1 --synonym 0 --antonym 0 --metadata LLM-pe
./measure_classification_density_all.sh --experiment ICRA-ablation-regular-queryability --type method --prompt_eng 1 --synonym 1 --antonym 1 --metadata gt-pe
