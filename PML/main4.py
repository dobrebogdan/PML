#TODO: Debug new approach, maybe use NN, maybe fewer words

import matplotlib.pyplot as plt
import numpy as np
import csv
from SVM_classifier import SVM_classifier
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
from german_lemmatizer import lemmatize
import spacy
import gensim
import train

mode = "test"
# dev or test


nlp = spacy.load('de_core_news_sm')


freq_words = []
l = 0
with open("freq_words.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        for col in row:
            col = col.replace("'", "").replace(" ", "")

            freq_words.append(col)
            l+= 1
            if l >= 2000:
                break
print(len(freq_words))
words_cnt = {}

def text_to_tokens(curr_str):
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
    good_tokens = []
    for token in tokens:
        #print(token, token.lemma_)
        token = str(token.lemma_)
        if token == "" or (not token[0].isalpha()):
            continue
        good_tokens.append(token)
    return good_tokens

nr = 0
def text_to_coords(model, curr_str):
    global nr
    nr += 1
    print("CONVERSION BEGIN")
    print(nr)
    tokens = text_to_tokens(curr_str)
    sum = []
    cnt = 0.0
    for i in range(0, len(tokens)):
        if sum == []:
            try:
                sum = model.wv[tokens[i]].copy()
                cnt += 1.0
            except:
                pass
        else:
            try:
                sum += model.wv[tokens[i]]
                cnt += 1.0
            except:
                pass
    sum = sum / cnt
    print("CONVERSION END")
    return sum

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
train_text = []
train_tokens = []
train_labels = []


with open("training.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        train_text.append(row[3])
        train_tokens.append(text_to_tokens((row[3])))
        row[1] = float(row[1])
        row[2] = float(row[2])
        #train_labels.append(np.float(row[0]))
        train_labels.append(str(row[1]) + "," + str(row[2]))

model = gensim.models.Word2Vec(train_tokens, size=300, min_count=1, workers=4)
train_data = []
for entry in train_text:
    train_data.append(text_to_coords(model, entry))
print("Model created")



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
            validation_data.append(text_to_coords(model, row[1]))
        else:
            validation_labels.append(str(row[1]) + "," + str(row[2]))
            validation_data.append(text_to_coords(model, row[3]))

svm_classifier = SVM_classifier(train_data, train_labels, validation_ids, validation_data, validation_labels)

# C 1000 gamma 1, error 1.094
svm_classifier.classify_tweets()

