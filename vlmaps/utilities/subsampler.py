
import argparse
import numpy
import re
import os
import numpy as np
from pathlib import Path
import math

class SearchResult:
    def __init__(self, search_string):
        self.search_string = search_string
        self.embedding = None
        self.gt = None
        self.semantic = None
        self.instance = None
        self.grid = None

    def validate(self):
        missing_fields = []
        if self.embedding is None:
            missing_fields.append("embedding")
        if self.gt is None:
            missing_fields.append("gt")
        if self.semantic is None:
            missing_fields.append("semantic")
        if self.instance is None:
            missing_fields.append("instance")
        if self.grid is None:
            missing_fields.append("grid")

        if missing_fields:
            print(f"Search string: {self.search_string}")
            print("Missing files for fields:", missing_fields)
            raise ValueError("One or more fields are missing a file.")
        elif len({self.embedding, self.gt, self.semantic, self.instance, self.grid}) < 5:
            raise ValueError("One or more fields have more than one file assigned.")
        else:
            print(f"{self.search_string}: Map loaded")

    def print_fields(self):
        print("Embedding:", self.embedding)
        print("GT:", self.gt)
        print("Semantic:", self.semantic)
        print("Instance:", self.instance)
        print("Grid:", self.grid)

def find_maps(directory):
    var1_values = set()  # Use a set to store unique variable names
    # Define the pattern to match files
    pattern = re.compile(r'(.+)_\w+\.data\.npy')

    # Iterate over files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the pattern
            match = pattern.match(file)
            if match:
                var1 = match.group(1)
                var1_values.add(var1)  # Add to set to keep unique variables

    return list(var1_values)  # Convert set to list before returning

def subsampleOpenscene(dir, out, subsampling, force):
    maps = find_maps(dir)
    for map in maps:
        file = search_files(dir, map)
        subsampleOpensceneFile(file, dir, out, subsampling, force)

def search_files(folder, search_string):
    found_files = SearchResult(search_string)

    # Traverse through all files and directories recursively
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            if search_string.lower() in file_name.lower():  # Check if search string is a substring of the file name
                file_path = os.path.join(root, file_name)
                if "embedding" in file_name and found_files.embedding is None:
                    found_files.embedding = file_path
                elif "gt" in file_name and found_files.gt is None:
                    found_files.gt = file_path
                elif "semantic" in file_name and found_files.semantic is None:
                    found_files.semantic = file_path
                elif "instance" in file_name and found_files.instance is None:
                    found_files.instance = file_path
                elif "grid" in file_name and found_files.grid is None:
                    found_files.grid = file_path
    found_files.validate()
    return found_files

def createDirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def createSubsamplingIndices(semantics, label, subsampling):
    label_indices = np.where(semantics == label)[0]
    non_label_indices = np.where(semantics != label)[0]
    label_cnt = label_indices.shape[0]
    num_to_pick = math.ceil(label_cnt * subsampling)
    if(num_to_pick < 200):
        num_to_pick = label_cnt

    pick_idx = np.random.choice(label_cnt, num_to_pick, replace=False)
    indices = np.concatenate((label_indices[pick_idx], non_label_indices))

    return indices

def subsampleOpensceneFile(file, input, out, subsampling, force):
    outfile_embedding = file.embedding.replace(input, out)
    outfile_grid = file.grid.replace(input, out)
    outfile_gt = file.gt.replace(input, out)
    outfile_instance = file.instance.replace(input, out)
    outfile_semantic = file.semantic.replace(input, out)

    if (Path(outfile_embedding).exists() and
        Path(outfile_grid).exists() and
        Path(outfile_gt).exists() and
        Path(outfile_instance).exists() and
        Path(outfile_semantic).exists()):
        print(f"Output exists (e.g. {outfile_embedding})")
        if force:
            print("Overwriting")
        else:
            print("skipping")
            return

    print("*"*80)
    embedding = np.load(file.embedding)
    grid = np.load(file.grid)
    gt = np.load(file.gt)
    instance = np.load(file.instance)
    semantic = np.load(file.semantic)

    print("from:")
    print(embedding.shape[0])
    print(grid.shape[1])
    print(gt.shape[0])
    print(instance.shape[0])
    print(semantic.shape[0])

    labels = np.unique(semantic)
    print(labels.size, "unique labels")

    ss_embedding = np.copy(embedding)
    ss_grid = np.copy(grid)
    ss_gt = np.copy(gt)
    ss_instance = np.copy(instance)
    ss_semantic = np.copy(semantic)

    for label in labels:
        indices = createSubsamplingIndices(ss_semantic, label, subsampling)

        ss_embedding = ss_embedding[indices, :]
        ss_grid = ss_grid[:, indices]
        ss_gt = ss_gt[indices]
        ss_instance = ss_instance[indices]
        ss_semantic = ss_semantic[indices]

    #debug
    print("to:")
    print(ss_embedding.shape[0], " => ", ss_embedding.shape[0]/embedding.shape[0])
    print(ss_grid.shape[1], " => ", ss_grid.shape[1]/grid.shape[1])
    print(ss_gt.shape[0], " => ", ss_gt.shape[0]/gt.shape[0])
    print(ss_instance.shape[0], " => ", ss_instance.shape[0]/instance.shape[0])
    print(ss_semantic.shape[0], " => ", ss_semantic.shape[0]/semantic.shape[0])

    print(np.unique(ss_semantic).size, "unique labels")

    # save
    np.save(outfile_embedding, ss_embedding)
    np.save(outfile_grid, ss_grid)
    np.save(outfile_gt, ss_gt)
    np.save(outfile_instance, ss_instance)
    np.save(outfile_semantic, ss_semantic)

def subsampleVLMaps(input, output, subsampling, force):
    for i in range(10):
        outfile = output + "/" + str(i) + ".data.npy"
        if Path(outfile).exists():
            print(f"Output file {outfile} exists.")
            if force:
                print("Overwriting")
            else:
                print("Skipping")
                continue
        print("*"*80)
        data = np.load(input + "/" + str(i) + ".data.npy")
        print("in shape", data.shape)
        semantic = data[:, 0, 0]
        instance = data[:, 1, 0]
        region = data[:, 2, 0]
        embedding = data[:, 3, :]

        print(semantic.shape)

        ss_embedding = np.copy(embedding)
        ss_instance = np.copy(instance)
        ss_region = np.copy(region)
        ss_semantic = np.copy(semantic)

        labels = np.unique(ss_semantic)
        print(labels.size, "unique labels")

        for label in labels:
            indices = createSubsamplingIndices(ss_semantic, label, subsampling)

            ss_embedding = ss_embedding[indices, :]
            ss_region = ss_region[indices]
            ss_instance = ss_instance[indices]
            ss_semantic = ss_semantic[indices]

        print("to:")
        print(ss_embedding.shape[0], " => ", ss_embedding.shape[0]/embedding.shape[0])
        print(ss_region.shape[0], " => ", ss_region.shape[0]/region.shape[0])
        print(ss_instance.shape[0], " => ", ss_instance.shape[0]/instance.shape[0])
        print(ss_semantic.shape[0], " => ", ss_semantic.shape[0]/semantic.shape[0])

        print(np.unique(ss_semantic).size, "unique labels")
        print(ss_embedding.shape)
        print(ss_region.shape)
        print(ss_instance.shape)
        print(ss_semantic.shape)

        len = ss_semantic.shape[0]
        embedding_size = ss_embedding.shape[1]

        # save
        out_data = np.zeros((len, 4, embedding_size))
        out_data[:, 0, :] = ss_semantic.reshape(-1,1)
        out_data[:, 1, :] = ss_instance.reshape(-1, 1)
        out_data[:, 2, :] = ss_region.reshape(-1, 1)
        out_data[:, 3, :] = ss_embedding

        print("out shape", out_data.shape)
        np.save(outfile, out_data)

def main():
    parser = argparse.ArgumentParser("./subsampler.py")
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=True,
        default=""
    )
    parser.add_argument(
        '--output', '-o',
        dest="output",
        type=str,
        required=True,
        default=""
    )
    parser.add_argument(
        '--subsampling', '-s',
        dest="subsampling",
        type=float,
        required=True,
        default=0
    )
    parser.add_argument(
        '--method', '-m',
        dest="method",
        type=str,
        required=False,
        default="openscene"
    )
    parser.add_argument(
        '--force', '-f',
        dest="force",
        action="store_true",
        required=False,
        default=False
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input
    output = FLAGS.output
    subsampling = FLAGS.subsampling
    method = FLAGS.method
    force = FLAGS.force

    print("="*80)
    print("="*80)
    print(f"Subsample: {input}\nto: {output}\nRatio: {subsampling}\nMethod: {method}")
    print("="*80)
    print("="*80)

    createDirs(output)
    if(method == "openscene"):
        subsampleOpenscene(input, output, subsampling, force)
    else:
        subsampleVLMaps(input, output, subsampling, force)


if __name__ == "__main__":
    main()
