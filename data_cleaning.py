from pandas import read_csv
from numpy import random

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

import os
import re
random.seed(7000)
home = os.getcwd()

corpus = read_csv(f"{home}/dataset.csv", encoding="latin-1")
stopwords = read_csv(f"{home}/stopword.csv", encoding="latin-1")
corpus['text_final'] = [low.lower() for low in corpus['text']]

#Remove any number
corpus['text_final'] = [re.sub("\d+", "", text_final) for text_final in corpus['text_final']]
#Remove string with <>
corpus['text_final'] = [re.sub("<\S+>", "", text_final) for text_final in corpus['text_final']]
#Remove all characters except alphabets
corpus['text_final'] = [re.sub("[^\w\s]", "", text_final) for text_final in corpus['text_final']]
#Replace multiple spaces with a space
corpus['text_final'] = [text_final.strip() for text_final in corpus['text_final']]
#Stemming ("memakai" -> "pakai")
stem = StemmerFactory().create_stemmer()
corpus['text_final'] = [stem.stem(tok) for tok in corpus['text_final']]
#Tokenize
corpus['text_final'] = [word_tokenize(text_final) for text_final in corpus['text_final']]
#Remove stopwords
for idx, text_final in enumerate(corpus['text_final']):
    finalized = []
    for te in text_final:
        if te not in stopwords['stopword'].values.tolist():
            finalized.append(te)
    corpus.loc[idx, 'text_final'] = str(finalized)

corpus.to_csv("clean.csv")