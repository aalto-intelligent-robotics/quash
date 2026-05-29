import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Tuple


def save_cache(hash_object: Any, store_object: Any) -> None:
    hash = get_hash(hash_object)
    cache_folder = "/home/user/code/vlmaps/cache"
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    with open(f"{cache_folder}/{hash}", "wb") as f:
        pickle.dump(store_object, f)

def load_cache(hash_object: Any) -> Tuple[bool, Any]:
    hash = get_hash(hash_object)
    cache_folder = "/home/user/code/vlmaps/cache"
    filepath = f"{cache_folder}/{hash}"
    if Path(filepath).exists():
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
            return True, obj
    else:
        return False, None

def get_hash(object: Any) -> str:
    return hashlib.sha256(pickle.dumps(object)).hexdigest()

if __name__ == "__main__":
    a = 1
    b = "string"
    c = np.eye(3)
    #save_cache((a, b, c))
    load_cache((a, b, c))

