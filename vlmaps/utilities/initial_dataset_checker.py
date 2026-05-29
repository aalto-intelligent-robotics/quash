import os

failure = False

scenes = ["5LpN3gDmAk7",
 "gTV8FGcVJC9",
 "jh4fc5c5qoQ",
 "JmbYfDe2QKZ",
 "mJXqzFtmKg4",
 "ur6pFq6Qu1A",
 "UwV83HsGsw3",
 "Vt2qJdWjCF2",
 "YmJkqBEsHnH"]

yaml_path = "../vlmaps/config/data_paths/default.yaml"

f = open(yaml_path, "r")
lines = f.readlines()
if(len(lines) < 2):
    print("Invalid configuration of file vlmaps/config/data_paths/default.yaml. Please see configuration instructions.")
    exit(1)

try:
    habitat_dir = lines[1].rstrip("\n").lstrip("habitat_scene_dir:").rstrip("\"").lstrip(" \"")
    out_dir = lines[2].lstrip("vlmaps_data_dir:").rstrip("\"").lstrip(" \"")
except:
    print("Invalid configuration of file vlmaps/config/data_paths/default.yaml. Please see configuration instructions.")
    exit(1)

for scene in scenes:
    dirpath = habitat_dir + "/" + scene
    dirfound = os.path.isdir(dirpath)
    if(not dirfound):
        print("Missing directory:", dirpath)
        failure = True

    glb = dirpath + "/" + scene + ".glb"
    glbfound = os.path.exists(glb)
    if(not glbfound):
        print("Missing .glb", glb)
        failure = True

    house = dirpath + "/" + scene + ".house"
    housefound = os.path.exists(house)
    if (not housefound):
        print("Missing .house", house)
        failure = True

    navmesh = dirpath + "/" + scene + ".navmesh"
    navmeshfound = os.path.exists(navmesh)
    if (not navmeshfound):
        print("Missing .navmesh", navmesh)
        failure = True

    ply = dirpath + "/" + scene + "_semantic.ply"
    plyfound = os.path.exists(ply)
    if (not plyfound):
        print("Missing .ply", ply)
        failure = True

if(not failure):
    print("dataset ok!")
    exit(0)
else:
    exit(1)
