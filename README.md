# QuASH: Using Natural-Language Heuristics to Query Visual-Language Maps

**Authors:** Anonymous
**Affiliation:** Anonymous

Embeddings from Visual-Language Models are increasingly utilized to represent semantics in robotic maps, offering an open-vocabulary scene understanding that surpasses traditional, limited labels. Embeddings enable on-demand querying by comparing embedded user text prompts to map embeddings via a similarity metric. To perform the task indicated in a query, the robot must determine the parts of the environment relevant to the query. Two key challenges remain: the map creation and the selection of relevant matches.
In map creation, embeddings are typically aggregated as their mean, which can alter the semantics, especially when the embeddings within a map cell are multimodally distributed. In match selection, comparison of the query to a single complementary query is often used, which may fail to capture the underlying complex distributions.

This paper proposes solutions to these challenges. First, we propose the geometric mean for aggregation, which alleviates the semantic drift. Second, we leverage natural-language synonyms and antonyms associated with the query within the embedding space, applying heuristics to estimate the language space relevant to the query, and use that to train a classifier to partition the environment into matches and non-matches.
We evaluate our methods through extensive experiments, querying both maps and standard image benchmarks. The results demonstrate increased queryability of maps and images. Our querying technique is agnostic to the representation and encoder used, and requires limited training.

## Installation

```bash
    # TODO
```

## Citation

TODO