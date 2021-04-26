# from Ass1.training_model import get_word_embedding
import pickle

import nltk
import re
import numpy as np

from Ass1.training_model import clean_word_list, get_word_embedding

nltk.download("nps_chat")
from nltk.corpus import nps_chat, webtext

sents_webtext = []
for fileid in webtext.fileids():
    sents_webtext += list(webtext.sents(fileid))


def remove_words(sent: list):
    from nltk.corpus import stopwords
    stop_words = stopwords.words("english")
    for i in range(len(sent)):
        # print(type(sent[i]))
        sent[i] = "" if sent[i] in stop_words else sent[i]
        sent[i] = re.sub(r"[^A-Za-z]", "", sent[i])
        if len(sent[i]) == 1:
            sent[i] = ""
    while "" in sent:
        sent.remove("")
    sent = [s.lower() for s in sent]
    # print(sent)
    return sent


sents_nps = []
for post in nps_chat.posts():
    sents_nps.append(remove_words(post))
while [] in sents_nps:
    sents_nps.remove([])
print(sents_nps[:5])
sents_webtext = [remove_words(s) for s in sents_webtext]
training_data = pickle.load(open("training_data.pkl", "rb"))
sents_training = []

for emo, context in training_data:
    sents_training.append(clean_word_list(context))
print(sents_training[:5])
sents = sents_nps + sents_webtext + sents_training
length_list = []
for d in sents:
    length_list.append(len(d))
seq_max_len = max(length_list)
print(len(sents))
wv, word_embedding = get_word_embedding(sents, 300, seq_max_len)
print(word_embedding.shape)
