# QuASH: Using Natural-Language Heuristics to Query Visual-Language Maps

**Authors:** Anonymous
**Affiliation:** Anonymous

Embeddings from Visual-Language Models are increasingly utilized to represent semantics in robotic maps, offering an open-vocabulary scene understanding that surpasses traditional, limited labels. Embeddings enable on-demand querying by comparing embedded user text prompts to map embeddings via a similarity metric. To perform the task indicated in a query, the robot must determine the parts of the environment relevant to the query. Two key challenges remain: the map creation and the selection of relevant matches.
In map creation, embeddings are typically aggregated as their mean, which can alter the semantics, especially when the embeddings within a map cell are multimodally distributed. In match selection, comparison of the query to a single complementary query is often used, which may fail to capture the underlying complex distributions.

This paper proposes solutions to these challenges. First, we propose the geometric mean for aggregation, which alleviates the semantic drift. Second, we leverage natural-language synonyms and antonyms associated with the query within the embedding space, applying heuristics to estimate the language space relevant to the query, and use that to train a classifier to partition the environment into matches and non-matches.
We evaluate our methods through extensive experiments, querying both maps and standard image benchmarks. The results demonstrate increased queryability of maps and images. Our querying technique is agnostic to the representation and encoder used, and requires limited training.

## Installation

There are two repositories "quash" and "vlmaps". "quash" hosts the method and the image benchmarks. "vlmaps" hosts the map benchmarks. Both repositories have self-contained Dockerfiles. Remember to set up the setup_environment.sh scripts to correct paths.

Running image benchmarks:
```bash
    cd quash
    make # build the docker
    source setup_env.sh # setup environment paths
    python run_benchmark.py -c config/coco.yaml # --help for the parametrization
    python run_benchmark.py -c config/pascal_context_459.yaml # --help for the parametrization
```

Running map benchmarks:
```bash
    cd vlmaps
    make # build the docker
    source setup_environment.sh # setup environment paths
    ./1_0_run_all_vlmaps.sh # build all maps
    cd scripts
    ./0_run_all_experiments.sh
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{pekkanen_2026_quash
    title={QuASH: Using Natural-Language Heuristics to Query Visual-Language Maps},
    author={Pekkanen, Matti and Verdoja, Francesco and Kyrki, Ville},
    booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
    publisher={IEEE},
    year={2026},
    month={June},
    address={Vienna, Austria},
    volume={},
    number={},
    pages={XX--YY},
}

```