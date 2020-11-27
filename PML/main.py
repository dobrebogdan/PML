import matplotlib.pyplot as plt
import numpy as np
import csv
from Knn import Knn_classifier
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
from german_lemmatizer import lemmatize
import spacy
nlp = spacy.load('de_core_news_sm')

freq_words = ['ich', 'der', 'de', 'isch', 'und', 'i', 'sein', 'ned', 'sich', 'e', 'u', 'nid']

def text_to_coords(curr_str):
    for i in range(0, len(curr_str)):
        if (not curr_str[i] == " ") and (not curr_str[i].isalpha()):
            #words_cnt[curr_str[i]] = words_cnt.get(curr_str[i], 0) + 1
            #words_corelation[curr_str[i]] = add_tuples(words_corelation.get(curr_str[i], (0.0, 0.0)), pair[0])
            curr_str = curr_str.replace(curr_str[i], " ")
    curr_str = curr_str.lower()
    tokens = nlp.tokenizer(curr_str)
    curr_words_cnt = {}
    for token in tokens:
        #print(token, token.lemma_)
        token = str(token.lemma_)
        if token == "" or (not token[0].isalpha()):
            continue
        curr_words_cnt[token] = curr_words_cnt.get(token, 0.0) + 1.0
    curr_coords = []
    sum = 0
    for i in range(0, len(freq_words)):
        t = curr_words_cnt.get(freq_words[i], 0)
        sum = sum + t
        curr_coords.append(t)

    if sum ==0:
        sum = 1
    for i in range(0, len(curr_coords)):
        curr_coords[i] /= sum
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
train_data_dict = {}
with open("training.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        train_data.append(text_to_coords(row[3]))
        row[1] = float(row[1])
        row[2] = float(row[2])
        train_labels.append(str(row[1]) + "," + str(row[2]))

        train_data_dict[(row[1], row[2])] = train_data_dict.get((row[1], row[2]), "") + " " + row[3]


print("LMAOOOO")
validation_data = []
validation_labels = []
with open("validation.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        validation_data.append(text_to_coords(row[3]))
        validation_labels.append(str(row[1]) + "," + str(row[2]))


train_data = train_data
train_labels = train_labels

validation_data = validation_data

print(validation_data)
validation_labels = validation_labels



knn_classifier = Knn_classifier(train_data, train_labels, validation_data, validation_labels)
knn_classifier.classify_tweets()

# sunt mai multe tweeturi de la aceleasi persoane. Cate? Este un fenomen raspandit?
