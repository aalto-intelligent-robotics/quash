#!/bin/bash

if [ -z $1 ]
then
    echo "Usage ./predict_all.sh <model=new/lora> <encoder=lseg/openseg>"
    exit 0
fi

if [ -z $2 ]
then
    echo "Usage ./predict_all.sh <model=new/lora> <encoder=lseg/openseg>"
    exit 0
fi

if [[ $2 == "lseg" ]]
then
    dim=512
else
    dim=768
fi

python predict_map.py --encoder $2 --dim $dim --path models/$1-46-$2 --maps 0 1 2 3 --max_tokens 6000
python predict_map.py --encoder $2 --dim $dim --path models/$1-03-$2 --maps 4 5 6 7 8 9 --max_tokens 6000

# Max tokens per map. Total max 5497
# Max width: 5497
# Max width: 657
# Max width: 991
# Max width: 1943
# Max width: 2148
# Max width: 3861
# Max width: 1989
# Max width: 2874
# Max width: 1824
# Max width: 528
