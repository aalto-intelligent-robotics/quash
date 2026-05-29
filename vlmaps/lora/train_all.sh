#!/bin/bash

epochs=250
its=-1
N=20
tokens=6000

if [ -z $1 ]
then
    echo "Usage ./train_all.sh <model=new/lora> <encoder=lseg/openseg>"
    exit 0
fi

if [ -z $2 ]
then
    echo "Usage ./train_all.sh <model=new/lora> <encoder=lseg/openseg>"
    exit 0
fi

if [[ $2 == "openseg" ]]
then
    dim=768
else
    dim=512
fi


if [[ $1 == "lora" ]]
then
    python train.py --train --name lora-03-$2 --dim $dim --maps 0 1 2 3 --epochs $epochs --intermediate $its --max_tokens $tokens --validate_all --encoder $2 --n $N
    python train.py --train --name lora-46-$2 --dim $dim --maps 4 5 6 --epochs $epochs --intermediate $its --max_tokens $tokens --validate_all --encoder $2 --n $N
    python train.py --train --name lora-79-$2 --dim $dim --maps 7 8 9 --epochs $epochs --intermediate $its --max_tokens $tokens --validate_all --encoder $2 --n $N
elif [[ $1 == "new" ]]
then
    python train.py --train --new --name new03-$2 --dim $dim --maps 0 1 2 3 --epochs $epochs --intermediate $its --validate_all --encoder $2 --n $N
    python train.py --train --new --name new46-$2 --dim $dim --maps 4 5 6 --epochs $epochs --intermediate $its --validate_all --encoder $2 --n $N
    python train.py --train --new --name new79-$2 --dim $dim --maps 7 8 9 --epochs $epochs --intermediate $its --validate_all --encoder $2 --n $N
else
    echo "Usage ./train_all.sh <model=new/lora>"
fi