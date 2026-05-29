import difflib
import ast
from tqdm import tqdm
import nltk
from nltk.corpus import words
import random
import numpy as np

nltk.download('words')
english_words = set(words.words())
ths = np.array(range(1, 10, 2))/10
num = 7

inp = "n"
while inp != "y":
    test_set = random.sample(english_words, 5)
    rest = english_words.difference(test_set)
    print(test_set)
    inp = input("ok?")
    if inp == "q":
        exit(0)

for th in tqdm(ths):
    res = {}
    for word in tqdm(test_set, leave=False):
        close_matches = difflib.get_close_matches(word, rest, n=num, cutoff=th)
        res[word] = close_matches

    print("Matches with threshold", th)
    for key in res:
        print(key, ":", res[key])
