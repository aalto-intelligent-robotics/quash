# Query distribution

Open-vocabulary image segmentation using LLMs.

## Running

#### Create docker
```bash
docker compose up -d --build
docker attach qd
./post_install.sh
```

#### Run
```bash
source setup_env.sh
source openai_key.sh
python run_single.py -i <image> -t <query> -c <"other"> -s <synonyms for query> -cs <antonyms for query>
```

- -i is the path to input image
- -t is for the query term used for baseline. Use that as X of the prompt
- -c is the complement category (i.e., "other") used for baseline.
- -s are the synonyms for -t, created using ChatGPT.
- -cs are the complement synonyms for -t, created using ChatGPT.