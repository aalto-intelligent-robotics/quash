#!/bin/bash
if [ -z $1 ]
then
    echo "usage: downloader.sh <download dir>"
    exit
else
    dir=$1
fi

echo $dir

cd ../dataset
python download_mp.py -o $dir --id 5LpN3gDmAk7 --task habitat
python download_mp.py -o $dir --id gTV8FGcVJC9 --task habitat
python download_mp.py -o $dir --id jh4fc5c5qoQ --task habitat
python download_mp.py -o $dir --id JmbYfDe2QKZ --task habitat
python download_mp.py -o $dir --id JmbYfDe2QKZ --task habitat
python download_mp.py -o $dir --id mJXqzFtmKg4 --task habitat
python download_mp.py -o $dir --id ur6pFq6Qu1A --task habitat
python download_mp.py -o $dir --id UwV83HsGsw3 --task habitat
python download_mp.py -o $dir --id Vt2qJdWjCF2 --task habitat
python download_mp.py -o $dir --id YmJkqBEsHnH --task habitat