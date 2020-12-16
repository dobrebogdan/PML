# tf idf + svm

import matplotlib.pyplot as plt
import numpy as np
import csv
from SVM_classifier import SVM_classifier
from nltk.corpus import stopwords
import nltk
import spacy
import train

mode = "dev"
# dev or test

#nltk.download("stopwords")

german_stop_words = stopwords.words('german')


nlp = spacy.load('de_core_news_sm')

freq_words = ['isch', 'de', 'au', 'i', 'ja', 'e', 'en', 'no', 'uf', 'bi', 'nid', 'd', 'ha', 'het', 'scho', 'vo', 'ned', 's', 'z',
              'oj', 'mer', 'nöd', 'bisch', 'han', 'hesch', 'mal', 'si', 'eifach', 'immer', 'u', 'gsi', 'dr', 'grad', 'a', 'mi', 'nei',
              'wer', 'di', 'guet', 'kei', 'jo', 'us', 'hani', 'haha', 'chli', 'ah', 'gut', 'jodel', 'me', 'mini', 'go', 'meh', 'weiss',
              'hend', 'm', 'ganz', 'mol', 'esch', 'min', 'chasch', 'cha', 'ds', 'hüt', 'wär', 'jed', 'w', 'wieso', 'bim', 'danke', 'nüt',
              'gha', 'gern', 'eh', 'sichern', 'gleich', 'nit', 'nie', 'lüt', 'do', 'gmacht', 'voll', 'dä', 'ond', 'öpper', 'git', 'o',
              'alli', 'alt', 'ig', 'mega', 'chan', 'tag', 'gseh', 'genau', 'dini', 'schaffen', 'em', 'glaub', 'lieben', 'kennen', 'gits',
              'ab', 'cho', 'schön', 'lang', 'richtig', 'frau', 'hed', 'aso', 'jahr', 'ide', 'all', 'gad', 'mis', 'paar', 'ferie', 'ou',
              'denken', 'sorry', 'halt', 'oh', 'ni', 'leider', 'abr', 'luege', 'wenns', 'is', 'muesch', 'öppis', 'mau', 'hei', 'gar',
              'moment', 'selber', 'fraue', 'wuche', 'bini', 'muess', 'gueti', 'ohni', 'bitte', 'würd', 'morn', 'ebe', 'mim', 'na', 'hett',
              'det', 'ech', 'problem', 'öper', 'sit', 'hets', 'ih', 'super', 'finden', 'ä', 'eu', 'geil', 'ischs', 'ke', 'ish', 'worde',
              'din', 'gseit', 'gaht', 'guete', 'hät', 'ne', 'heisst', 'weisch', 'bett', 'be', 'recht', 'id', 'fründ', 'ez', 'eher', 'seit',
              'erst', 'nume', 'echt', 'chunt', 'leben', 'wür', 'hahaha', 'odr', 'zug', 'fründin', 'laufen', 'essen', 'wennd', 'kik', 'eis',
              'wenig', 'nüm', 'eini', 'druf', 'schreiben', 'sex', 'öpis', 'isches', 'mach', 'helfen', 'jaa', 'merci', 'gseht', 'chunnt',
              'jhj', 'sii', 'mr', 'ufem', 'schlimm', 'use', 'gah', 'geben', 'kolleg', 'has', 'eigentlich', 'auto', 'ga', 'passieren', 'wellen',
              'sogar', 'ma', 'frag', 'liebi', 'morge', 'lust', 'tun', 'woni', 'sache', 'glück', 'wiso', 'bier', 'klar']


words_cnt = {}

def text_to_coords(curr_str):
    emojis_count = 0.0
    for i in range(0, len(curr_str)):
        if (not curr_str[i] == " ") and (not curr_str[i].isalpha()) \
                and (not curr_str[i] in [",", ".", ";", "!", "?", ":", "(", ")", "[", "]", "{", "}","-","+"]):
            emojis_count += 1.0
    for i in range(0, len(curr_str)):
        if (not curr_str[i] == " ") and (not curr_str[i].isalpha()):
            curr_str = curr_str.replace(curr_str[i], " ")
    curr_str = curr_str.lower()
    tokens = nlp.tokenizer(curr_str)
    curr_words_cnt = {}
    for token in tokens:
        #print(token, token.lemma_)
        token = str(token.lemma_)
        if token == "" or (not token[0].isalpha()):
            continue
        words_cnt[token] = words_cnt.get(token, 0) + 1
        curr_words_cnt[token] = curr_words_cnt.get(token, 0.0) + 1.0
    curr_coords = []
    sum = 0.0
    for i in range(0, len(freq_words)):
        t = curr_words_cnt.get(freq_words[i], 0.0)
        sum = sum + t
        curr_coords.append(t)
    #curr_coords.append(emojis_cont)
    #sum = sum + emojis_cont
    if sum == 0.0:
        sum = 1.0
    for i in range(0, len(curr_coords)):
        curr_coords[i] /= sum
        curr_coords[i] = np.float32(curr_coords[i])
    return curr_coords


def add_tuples(a, b):
    return (a[0] + b[0], a[1] + b[1])

def div_tuple(a, d):
    return (a[0]/d, a[1]/d)

def useful_data(coords):
    for i in range(0, len(coords)):
        if coords[i] > 0:
            return True
    return False
# Press the green button in the gutter to run the script.
train_data = []
train_labels = []


with open("training.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        train_data.append(text_to_coords((row[3])))
        row[1] = float(row[1])
        row[2] = float(row[2])
        train_labels.append(np.float(row[0]))
        #train_labels.append(str(row[1]) + "," + str(row[2]))

train_data = np.array(train_data)
train_labels = np.array(train_labels)
words_cnt_lst = []
for item in words_cnt.items():
    words_cnt_lst.append((item[1], item[0]))

words_cnt_lst = sorted(words_cnt_lst, reverse=True)
with open("freq_words.txt", "w") as f:
    for pair in words_cnt_lst:
        f.write("'" + str(nlp.tokenizer(pair[1])[0].lemma_) + "', ")

validation_data = []
validation_labels = []
validation_ids = []

validation_file = "test.txt"
if mode == "dev":
    validation_file = "validation.txt"


with open(validation_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        validation_ids.append(row[0])
        if mode == "test":
            validation_data.append(text_to_coords(row[1]))
        else:
            validation_labels.append(str(row[1]) + "," + str(row[2]))
            validation_data.append(text_to_coords(row[3]))

print("STARTED TRAINING")
svm_classifier = SVM_classifier(train_data, train_labels, validation_ids, validation_data, validation_labels)

# C 1000 gamma 1, error 1.094
svm_classifier.classify_tweets()

# TODO: implement neural network classifier and use it
