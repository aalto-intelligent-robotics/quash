import argparse

from clip_embeddings.embeddingCreator import EmbeddingCreator
import numpy as np
from tqdm import tqdm

def distance(a, b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./embedding_means_to_list_comparator.py")
    parser.add_argument(
        '--a', '-a',
        dest="a",
        type=str
    )
    parser.add_argument(
        '--b', '-b',
        dest="b",
        type=str
    )
    parser.add_argument(
        '--out', '-o',
        dest="out",
        type=str,
        required=False,
        default=""
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Use cpu or cuda.",
    )
    FLAGS, unparsed = parser.parse_known_args()
    a = FLAGS.a
    b = FLAGS.b
    device = FLAGS.device
    out = FLAGS.out

    writeOut = False
    if(out):
        writeOut = True

    data = {}

    creator = EmbeddingCreator(device)

    f = open(a, "r")
    lines = f.readlines()
    l = len(lines)
    embeddings_a = np.zeros((l,512))
    idx = 0
    for line_a in tqdm(lines):
        line_a = line_a.rstrip().lstrip()
        query_a = creator.get_text_embedding(line_a)
        embeddings_a[idx, :] = query_a
    f.close()

    mean = np.mean(embeddings_a, axis=0)
    print(mean.shape)

    f = open(b, "r")
    lines = f.readlines()
    l = len(lines)
    embeddings_a = np.zeros((l, 512))
    idx = 0
    for line_b in tqdm(lines):
        line_b = line_b.rstrip().lstrip()
        query_b = creator.get_text_embedding(line_b)
        d = distance(query_b, mean)
        data[line_b] = d
    f.close()


    maxd = -2
    maxc = ""
    for key in data:
        d = data[key]
        print(f'{key:<15}', ":", d)
        if(d > maxd):
            maxd = d
            maxc = key

    print("Most related to term to list a is", f'{maxc:<15}', "with distance", f'{maxd:<15}')
