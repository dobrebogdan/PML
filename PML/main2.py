import matplotlib.pyplot as plt
import numpy as np
import csv
from NN_classifier import NN_classifier
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
from german_lemmatizer import lemmatize
import spacy
import train

mode = "dev"
# dev or test


nlp = spacy.load('de_core_news_sm')


freq_words = ['ich', 'der', 'und', 'isch', 'de', 'sich', 'so', 'aber', 'au', 'i', 'mein', 'du',
              'wo', 'was', 'mit', 'ja', 'e', 'en', 'uf', 'no', 'im','wie','wenn','am','oder,',
              'bi','nid', 'd', 'ha', 'für', 'het', 'scho', 'ned', 'vo', 's', 'in', 'denn', 'z', 'mer',
              'oj', 'nöd', 'bisch', 'han', 'hesch', 'machen', 'nur', 'da', 'mal', 'si', 'eifach', 'immer',
              'meinen', 'zum', 'u', 'gsi', 'dr', 'werden', 'grad', 'a', 'dass']
"""

              'als','mi', 'nei', 'di', 'wer', 'guet', 'ein', 'kei', 'jo', 'zu','jetzt', 'us', 'hani',
              'haha', 'jodel', 'ah', 'mini', 'gut', 'me', 'chli', 'all', 'wollen', 'go', 'also', 'viel',
              'meh', 'hend', 'weiss', 'm', 'mol', 'ganz', 'vom', 'esch', 'min',
              'vor', 'chasch', 'ds', 'cha', 'hüt', 'wär', 'jed', 'wieso', 'w', 'bim', 'danke', 'nüt',
              'gha', 'gern', 'sichern', 'eh', 'wieder', 'gleich', 'nach', 'bis', 'nit', 'nie', 'lüt',
              'do', 'nicht', 'haben', 'gmacht', 'dä', 'an', 'voll', 'ond', 'o', 'öpper', 'git', 'alli',
              'mega', 'sehr', 'genau', 'alt', 'über', 'tag', 'ig', 'dini', 'chan', 'gseh', 'schaffen',
              'glaub', 'em', 'kennen', 'mein', 'lieben', 'ab', 'um', 'gits', 'cho', 'schön', 'lang', 'hed',
              'richtig', 'können', 'aso', 'frau', 'jahr', 'ide', 'gad', 'all', 'paar',
              'mis', 'denken', 'kein', 'ou', 'ander', 'sorry', 'halt', 'ferie', 'weg', 'abr', 'ni', 'wenns',
              'oh', 'luege', 'leider', 'is', 'öppis', 'moment', 'hei', 'mau', 'gar', 'muesch', 'wuche',
              'selber', 'bini', 'fraue', 'bitte', 'ohni', 'gueti', 'muess', 'morn', 'ebe', 'würd', 'mim',
              'hett', 'det', 'problem', 'hets', 'na', 'sit', 'öper', 'ech', 'super', 'finden', 'ob', 'ih',
              'man', 'ä', 'ke', 'geil', 'gaht', 'ish', 'eu', 'din', 'ischs', 'aus', 'gseit', 'hät', 'guete', 'worde', 'be', 'ne', 'weisch']
"""


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

# bidirectional dictionary
labels_dict = {}

curr_id = 0
with open("training.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        train_data.append(text_to_coords((row[3])))
        row[1] = float(row[1])
        row[2] = float(row[2])
        raw_label = str(row[1]) + "," + str(row[2])
        if raw_label in labels_dict.keys():
            train_labels.append(labels_dict[raw_label])
        else:
            labels_dict[raw_label] = curr_id
            labels_dict[curr_id] = raw_label
            train_labels.append(curr_id)
            curr_id += 1

train_data = np.array(train_data)
train_labels = np.array(train_labels)
words_cnt_lst = []
for item in words_cnt.items():
    words_cnt_lst.append((item[1], item[0]))

words_cnt_lst = sorted(words_cnt_lst, reverse=True)
with open("freq_words.txt", "w") as f:
    for pair in words_cnt_lst:
        f.write("'" + str(nlp.tokenizer(pair[1])[0].lemma_) + "', ")

# TRAINING
train.train_NN(train_data, train_labels)


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
            validation_labels.append(labels_dict[str(row[1]) + "," + str(row[2])])
            validation_data.append(text_to_coords(row[3]))

nn_classifier = NN_classifier(train_data, train_labels, validation_ids, np.array(validation_data), validation_labels, labels_dict)

print("STARTED CLASSIFYING")
nn_classifier.classify_tweets()

# TODO: implement neural network classifier and use it
