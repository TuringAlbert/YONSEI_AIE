import nltk
import gensim.downloader
from nltk.corpus import wordnet
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
# import openai
import os
from tqdm import tqdm

# Download WordNet corpus
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# api_key
# openai.api_key = 'Your API KEY'

def is_adjective_or_noun(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        if synset.pos() == 'a': #synset.pos() 의미론적으로 연결된 단어 집합)의 품사를 반환 'a' = adjective
            return "adjective"
        elif synset.pos() == 'n':
            return "noun"
    return False

# word = ['beautiful','dog','with']
word = 'dog'

if is_adjective_or_noun(word) == "adjective":
    print(f"{word} is an adjective.")
elif is_adjective_or_noun(word) == "noun":
    print(f"{word} is an noun.")
else:
    print(f"{word} is neither an adjective nor noun.")

model = gensim.downloader.load('glove-twitter-200')



target_appearance = ["girl", "hair"]
similar_words = model.most_similar(positive=target_appearance, topn=100)

for word, similarity in similar_words:
    print(word, is_adjective_or_noun(word), similarity)


