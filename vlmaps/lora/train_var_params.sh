#!/bin/bash

epochs=200
its=100

CUDA_VISIBLE_DEVICES=0 python train.py --train --name lora3-200-03-512-n100 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all --max_tokens 512  --n 100
CUDA_VISIBLE_DEVICES=0 python train.py --train --name lora3-200-03-512-n20 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all --max_tokens 512  --n 20
CUDA_VISIBLE_DEVICES=0 python train.py --train --name lora3-200-03-512-n5 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all --max_tokens 512  --n 5
CUDA_VISIBLE_DEVICES=0 python train.py --train --name lora3-200-03-512-n5 --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all --max_tokens 512  --n 1
CUDA_VISIBLE_DEVICES=0 python train.py --train --name lora3-200-03-6k-n100 --maps 0 1 2 3 --epochs $epochs --intermediate $its  --validate_all --max_tokens 6000 --n 100
CUDA_VISIBLE_DEVICES=0 python train.py --train --name lora3-200-03-6k-n20 --maps 0 1 2 3 --epochs $epochs --intermediate $its  --validate_all --max_tokens 6000 --n 20
