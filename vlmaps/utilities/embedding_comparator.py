import argparse

from clip_embeddings.embeddingCreator import EmbeddingCreator
import numpy as np
from tqdm import tqdm

def distance(a, b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./embedding_comparator.py")
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
        '--file', '-f',
        dest="file",
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
    file = FLAGS.file
    device = FLAGS.device
    out = FLAGS.out

    writeOut = False
    if(out):
        writeOut = True

    data = {}

    creator = EmbeddingCreator(device)
    if(file and not a and not b):
        f = open(file, "r")
        lines = f.readlines()
        for line_a in tqdm(lines, disable=not writeOut):
            for line_b in lines:
                line_a = line_a.rstrip().lstrip()
                line_b = line_b.rstrip().lstrip()
                if(line_a == line_b):
                    continue
                query_a = creator.get_text_embedding(line_a)
                query_b = creator.get_text_embedding(line_b)
                d = distance(query_a, query_b)
                #data[(line_a, line_b)] = d
                if(not line_a in data):
                    data[line_a] = {}
                data[line_a].update({line_b: d})

                if(out):
                    fo = open(out, "a")
                    fo.write(line_a + "; " + line_b + "; " + str(d) + "\n")
                    fo.close()
                else:
                    print(line_a + " - " + line_b + ": " + str(d))
        f.close()

        for key in data:
            comparisons = data[key]
            maxd = -2
            maxc = ""
            for comparison in comparisons:
                d = comparisons[comparison]
                if(d > maxd):
                    maxd = d
                    maxc = comparison

            print("Most related to term:",
                  f'{key:<15}', "is", f'{maxc:<15}', "with distance", f'{maxd:<15}')
    if(a and b):
        query_a = creator.get_text_embedding(a)
        query_b = creator.get_text_embedding(b)
        d = distance(query_a, query_b)
        print("Distance between", a, "and", b, "=", d)
