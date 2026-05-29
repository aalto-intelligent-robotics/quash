#!/bin/bash

epochs=200
its=100

CUDA_VISIBLE_DEVICES=1 python train.py --train --name lora3-200-03-6k-n5 --maps 0 1 2 3 --epochs $epochs --intermediate $its  --validate_all --max_tokens 6000 --n 5
CUDA_VISIBLE_DEVICES=1 python train.py --train --name lora3-200-03-6k-n1 --maps 0 1 2 3 --epochs $epochs --intermediate $its  --validate_all --max_tokens 6000 --n 1
CUDA_VISIBLE_DEVICES=1 python train.py --new --train --name new3-200-03-n100 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all  --n 100
CUDA_VISIBLE_DEVICES=1 python train.py --new --train --name new3-200-03-n20 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all  --n 20
CUDA_VISIBLE_DEVICES=1 python train.py --new --train --name new3-200-03-n5 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all  --n 5
CUDA_VISIBLE_DEVICES=1 python train.py --new --train --name new3-200-03-n5 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all  --n 1