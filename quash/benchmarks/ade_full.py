from benchmarks.dataset import Dataset
import numpy as np
import typing
from typing import List, Tuple, Any, Dict
from PIL import Image
import pickle
import pandas as pd
import json
import os
from methods.common.tasktype import TaskType


class ADEFull(Dataset):
    def __init__(self, config_file: str, task: TaskType) -> None:
        super().__init__(config_file, task)
        self.data_dir = self.config.get_config_mandatory("data_dir", str)
        self.set_used_classes()
        self.dataset_name = self.config.get_config_mandatory("dataset", str)
        raw_cache = self.config.get_config("cache")
        self.read_cache = False
        self.write_cache = False
        if raw_cache is not None:
            if "r" in raw_cache:
                self.read_cache = True
            if "w" in raw_cache:
                self.write_cache = True

        # load info
        with open(self.data_dir + "/index_ade20k.pkl", "rb") as f:
            data = pickle.load(f)

        # 'filename': 'array of length N=27574 with the image file names',
        # 'folder': 'array of length N with the image folder names.',
        # 'scene': 'array of length N providing the scene name
        #       (same classes as the Places database) for each image.',
        # 'objectIsPart': 'array of size [C,N] counting how many times an object is a
        #       part in each image. objectIsPart[c,i]=m if in image i object class c
        #       is a part of another object m times. For objects, objectIsPart[c,i]=0,
        #       and for parts we will find: objectIsPart[c,i] = objectPresence(c,i)',
        # 'objectPresence': 'array of size [C, N] with the object counts per image.
        #       objectPresence(c,i)=n if in image i there are n instances of object
        #       class c.',
        # 'objectcounts': 'array of length C with the number of instances
        #       for each object class.',
        # 'objectnames': 'array of length C with the object class names.',
        # 'proportionClassIsPart': 'array of length C with the proportion of times that
        #       class c behaves as a part. If proportionClassIsPart[c]=0 then it means
        #       that this is a main object (e.g., car, chair, ...). See bellow for a
        #       discussion on the utility of this variable.',
        # 'wordnet_found': 'array of length C. It indicates if the objectname was found
        #       in Wordnet.',
        # 'wordnet_level1': 'list of length C. WordNet associated.',
        # 'wordnet_synset': 'list of length C. WordNet synset for each object name.
        #       Shows the full hierarchy separated by .',
        # 'wordnet_hypernym': 'list of length C. WordNet hypernyms for each
        #       object name.',
        # 'wordnet_gloss': 'list of length C. WordNet definition.',
        # 'wordnet_synonyms': 'list of length C. Synonyms for the WordNet definition.',
        # 'wordnet_frequency': 'array of length C. How many times each wordnet appears'
        self.filename: list[str] = data["filename"]

        self.folder: list[str] = data["folder"]
        self.max_items = len(self.filename)
        self.class_data = pd.read_csv(
            self.data_dir + "/objects.txt", delimiter="\t", encoding="latin"
        )
        self.names: Dict[int, str] = dict(
            zip(self.class_data[" Name index "], self.class_data["Wordnet name "])
        )

        fixed_names = {}
        for key, name in self.names.items():
            if "," in name:
                parts = name.split(",")
                fixed_names[key] = parts[0]
            else:
                fixed_names[key] = name
        self.names = fixed_names

    def get_categories(self) -> typing.Dict[int, str]:
        return self.names

    def safe_json(self, file_path):
        encodings_to_try = ["utf-8", "iso-8859-1", "latin-1"]

        for encoding in encodings_to_try:
            try:
                # Try to open and read the file with the current encoding
                with open(file_path, "r", encoding=encoding, errors="replace") as file:
                    file_content = file.read()

                # Attempt to parse the JSON content
                json_data = json.loads(file_content)
                # print(f"Successfully read the file using {encoding} encoding.")
                return json_data

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # If a decoding error occurs, move to the next encoding
                print(f"Failed to read JSON with {encoding}: {e}")

            except OSError:
                # Handle other file-related errors (e.g., file not found)
                # print(f"Error opening file: {e}")
                return None

        # If all encodings fail
        print("All encoding attempts failed.")
        return None

    def _create_item(self, overall_index) -> List[Tuple[Any, ...]]:
        filename = self.filename[overall_index]
        folder = self.folder[overall_index].replace("ADE20K_2021_17_01", "")
        basepath = self.data_dir + folder + "/"

        attr_file_name = filename.replace(".jpg", ".json")
        image_path = basepath + filename
        json_path = basepath + attr_file_name

        # Check cache
        pickle_path = basepath + filename.replace(
            ".jpg", "_" + self.dataset_name + ".pkl"
        )
        if self.read_cache and os.path.exists(pickle_path):
            with open(pickle_path, "rb") as file:
                output = pickle.load(file)
                return output

        json_file = self.safe_json(json_path)
        assert json_file is not None
        data = json_file["annotation"]
        objects = data["object"]

        idxs = {}
        for i in range(len(objects)):
            object = objects[i]
            name_ndx = object["name_ndx"]
            name = object["raw_name"]
            if not self.check_class(name):
                if name not in self.ignored_classes:
                    self.ignored_classes.append((name_ndx, name))
                continue
            if name_ndx not in idxs:
                idxs[name_ndx] = [i]
            else:
                labels = idxs[name_ndx]
                labels.append(i)
                idxs[name_ndx] = labels

        output = []
        for idx in idxs:
            obj_ids = idxs[idx]
            mask = None
            for obj_id in obj_ids:
                object = objects[obj_id]
                name_ndx = object["name_ndx"]
                name = object["raw_name"]
                mask_path = object["instance_mask"]
                t_mask = Image.open(basepath + mask_path)
                if mask is None:
                    mask = np.array(t_mask)
                else:
                    mask += np.array(t_mask)
            # FIXME:
            item = (name, idx, mask, image_path)
            output.append(item)

        if self.write_cache:
            with open(pickle_path, "wb") as file:
                pickle.dump(output, file)

        return output

    # TODO
    def get_progress_str(self) -> str:
        return ""
