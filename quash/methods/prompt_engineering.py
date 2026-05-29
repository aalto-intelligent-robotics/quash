from typing import Callable
import numpy as np

# Directly from vlmaps!
multiple_templates = [
    "There is {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "I took a picture of of {}.",
    "I took a picture of of my {}.",  # itap: I took a picture of
    "I took a picture of of the {}.",
    "a photo of {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {}.",
    "a good photo of the {}.",
    "a bad photo of {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {}.",
    "a bright photo of the {}.",
    "a dark photo of {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def get_prompt_engineered_queries(query: str):
    mul_tmp = multiple_templates.copy()
    return [x.format(query) for x in mul_tmp]


def get_engineered_embeddings(query: str, embedding_fun: Callable):
    prompts = get_prompt_engineered_queries(query)
    embeddings_list = []
    for p in prompts:
        embeddings_list.append(embedding_fun(p))
    return np.stack(embeddings_list, axis=1)


def get_engineered_mean_embedding(query: str, embedding_fun: Callable):
    embeddings = get_engineered_embeddings(query, embedding_fun)
    return np.mean(embeddings, axis=1)
