import datasets
from tqdm import tqdm
from pprint import pprint
import numpy as np
import json

full_path = "/home/user/data/datasets/Datasets_HSN/ADE20k/ade.hf"
ds = datasets.load_from_disk(full_path)
ds = ds["train"]

ids = []
names = []
name_to_id = {}
id_to_name = {}

updates = 0
cnt = 0
lim = 10000

for d in tqdm(ds, leave=False):
    objs = d["objects"]
    for obj in objs:
        n = obj["raw_name"]
        i = obj["name_ndx"]
        if n not in names:
            names.append(n)
            updates += 1
        if i not in ids:
            ids.append(i)
            updates += 1
        if n not in name_to_id:
            name_to_id[n] = []
            updates += 1
        nids = name_to_id[n]
        if i not in nids:
            nids.append(i)
            name_to_id[n] = nids
            updates += 1
        if i not in id_to_name:
            id_to_name[i] = []
            updates += 1
        idns = id_to_name[i]
        if n not in idns:
            idns.append(n)
            id_to_name[i] = idns
            updates += 1

        cnt += 1
        if cnt == lim:
            print(updates, "in last", cnt ,"scans")
            # if updates == 0:
            #     break
            cnt = 0
            updates = 0


names = np.array(names)
names = np.unique(names)
print(names)
print("*"*80)

ids = np.array(ids)
ids = np.unique(ids)
print(ids)
print("*"*80)

pprint(name_to_id)
print("*"*80)

pprint(id_to_name)
print("*"*80)

dir = "/home/user/<path/to/quash>/benchmarks/ade/"
with open(dir + "name_to_id.json", "w") as f:
    json.dump(name_to_id, f)

with open(dir + "id_to_name.json", "w") as f:
    json.dump(id_to_name, f)