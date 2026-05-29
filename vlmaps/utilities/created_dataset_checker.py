import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", "-v", type=float, help="Voxel size", required=True)
args = parser.parse_args()
voxel_size = args.voxel_size

failure = False

scenes = [
    "5LpN3gDmAk7_1",
    "gTV8FGcVJC9_1",
    "jh4fc5c5qoQ_1",
    "JmbYfDe2QKZ_1",
    "JmbYfDe2QKZ_2",
    "mJXqzFtmKg4_1",
    "ur6pFq6Qu1A_1",
    "UwV83HsGsw3_1",
    "Vt2qJdWjCF2_1",
    "YmJkqBEsHnH_1",
]

encoders = ["lseg", "openseg"]

yaml_path = "../vlmaps/config/data_paths/default.yaml"
yaml_path = "/home/user/code/vlmaps/vlmaps/config/data_paths/default.yaml"

f = open(yaml_path, "r")
lines = f.readlines()
if len(lines) < 3:
    print(
        "Invalid configuration of file vlmaps/config/data_paths/default.yaml. \
        Please see configuration instructions."
    )
    exit(1)

try:
    habitat_found = False
    habitat_dir = ""
    for line in lines:
        if "habitat_scene_dir" in line:
            habitat_dir = (
                lines[1]
                .replace("habitat_scene_dir:", "")
                .strip(" ")
                .strip("\n")
                .strip('"')
            )
            habitat_found = True
            break
    if not habitat_found:
        print("Missing habitat_scene_dir in config")
        exit(1)

    out_found = False
    out_dir = ""
    for line in lines:
        if "vlmaps_data_dir" in line:
            out_dir = (
                lines[2]
                .replace("vlmaps_data_dir:", "")
                .strip(" ")
                .strip("\n")
                .strip('"')
                + "vlmaps_dataset"
            )
            out_found = True
    if not out_found:
        print("Missing vlmaps_data_dir in config")
        exit(1)

    map_prefix_found = False
    map_prefix = ""
    for line in lines:
        if "map_prefix" in line:
            map_prefix = lines[3].replace("map_prefix:", "")
            map_prefix = re.sub(r"#(.)*$", "", map_prefix)
            map_prefix = map_prefix.strip(" ").strip("\n").strip('"')
            map_prefix_found = True
    if not map_prefix_found:
        print("Missing map_prefix in config")
        exit(1)
except Exception:
    print(
        "Invalid configuration of file vlmaps/config/data_paths/default.yaml. \
            Please see configuration instructions."
    )
    exit(1)

outpath = out_dir
outfound = os.path.isdir(outpath)
if not outfound:
    print("Missing directory:", outpath)
    exit(1)

for scene in scenes:
    dirpath = out_dir + "/" + scene
    dirfound = os.path.isdir(dirpath)
    if not dirfound:
        print("Missing directory:", dirpath)
        failure = True

    depth = dirpath + "/" + "depth"
    depthfound = os.path.isdir(depth)
    if not depthfound:
        print("Missing depth folder", depth)
        failure = True

    regions = dirpath + "/" + "regions"
    regionsfound = os.path.isdir(regions)
    if not regionsfound:
        print("Missing regions folder", regions)
        failure = True

    semantic = dirpath + "/" + "semantic"
    semanticfound = os.path.isdir(semantic)
    if not semanticfound:
        print("Missing semantic folder", semantic)
        failure = True

    rgb = dirpath + "/" + "rgb"
    rgbfound = os.path.isdir(rgb)
    if not rgbfound:
        print("Missing rgb folder", rgb)
        failure = True

    semantic_blur = dirpath + "/" + "semantic-blur"
    semantic_blurfound = os.path.isdir(semantic_blur)
    if not semantic_blurfound:
        print("Missing semantic-blur folder", semantic_blur)
        failure = True

    vlmap = dirpath + "/" + "vlmap"
    vlmapfound = os.path.isdir(vlmap)
    if not vlmapfound:
        print("Missing vlmap folder", vlmap)
        failure = True

    for encoder in encoders:
        prefix = map_prefix + "-" + encoder

        map = dirpath + "/" + "vlmap" + f"/{prefix}-{voxel_size}.h5df"
        mapfound = os.path.exists(map)
        if not mapfound:
            print("Missing map", map)
            failure = True

        postprocess = dirpath + "/" + "vlmap" + f"/{prefix}-postprocessed-{voxel_size}.h5df"
        postprocessfound = os.path.exists(postprocess)
        if not postprocessfound:
            print("Missing postprocessed map", postprocess)
            failure = True

        predict = dirpath + "/" + "vlmap" + f"/{prefix}-predicted-{voxel_size}.h5df"
        predictfound = os.path.exists(predict)
        if not predictfound:
            print("Missing predicted map", predict)
            failure = True

        predicted_postprocessed = (
            dirpath + "/" + "vlmap" + f"/{prefix}-predicted-postprocessed-{voxel_size}.h5df"
        )
        predicted_postprocessedfound = os.path.exists(predicted_postprocessed)
        if not predicted_postprocessedfound:
            print("Missing predicted postprocesseded map", predicted_postprocessed)
            failure = True

        instances = dirpath + "/" + "vlmap" + f"/{prefix}-instances-{voxel_size}.h5df"
        instancesfound = os.path.exists(instances)
        if not instancesfound:
            print("Missing instances map", instances)
            failure = True

        ourinstances = dirpath + "/" + "vlmap" + f"/{prefix}-our-instances-{voxel_size}.h5df"
        ourinstancesfound = os.path.exists(ourinstances)
        if not ourinstancesfound:
            print("Missing our instances map", ourinstances)
            failure = True


if not failure:
    print("dataset ok!")
    exit(0)
else:
    exit(1)
