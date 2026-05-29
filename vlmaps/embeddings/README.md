# Get CLIP embedding

This is a simple Python script that returns various embeddings for a given text or image.

Currently CLIP and Flava are supported.

The embedding is printed in the terminal and can also be written to a binary file.

## Run

Run `python get_embeddings.py` or `python -m embeddings.get_embeddings`.

```
python get_embeddings.py --network <clip/flava> --model_name <model name (ViT-B/32)> --text/--image <query> (--device <cpu/cuda> --embedding_file embedding.bin)
```

usage: get_embeddings.py [-h] (--text TEXT | --image_path IMAGE_PATH) [--embedding_file EMBEDDING_FILE] [--device {cpu,cuda}]
                         --network {clip,flava,tbd} --model_name MODEL_NAME

optional arguments:
  -h, --help            show this help message and exit
  --text TEXT           Text query.
  --image_path IMAGE_PATH
                        Path to image used as a query.
  --embedding_file EMBEDDING_FILE
                        File in which to save the embedding in binary format. If empty, the embedding is not saved to a file.
  --device {cpu,cuda}   Use cpu or cuda.
  --network {clip,flava,tbd}, -n {clip,flava,tbd}
                        Which NN model to use
  --model_name MODEL_NAME, -m MODEL_NAME
                        Path to model

