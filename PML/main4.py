import csv
from SVM_classifier import SVM_classifier

import spacy
import gensim


mode = "test"
# dev or test

# ERROR: 1.17 with lemmatizer
# 1.17 without lemmatizer...

# ERROR is 0.92 with training word embeddings at 100 epochs
# ERROR is 0.90 without window and negatives

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
    # tokens = word_tokenize(curr_str)
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

# SOME RESULTS

# no window, no negative = 0.910
# window  1, no negative = 0.934
# window  5, no negative = 0.903
# window 10, no negative = 0.903       go with this
# window 20. no negative = 0.904
# window 10, negative 10 = 0.905
# window 10, negative 20 = 0.910
# window 20, negative 10 = 0.903
# window 20, negative 20 = 0.904

# alpha = 0.01, min_alpha = 0.001 => 0.89
# alpha = 0.01, min_alpha = 0.01 sample = 6e-5=> 0.88
# alpha = 0.1, min_alpha = 0.1 sample = 6e-5 => 0.95
# alpha = 0.01, min_alpha = 0.01, sample = 1e-3 => 0.89
# alpha = 0.01, min_alpha = 0.01, sample = 1e-4 => 0.89
# alpha = 0.01, min_alpha = 0.01 sample = 5e-5=> 0.88
model = gensim.models.Word2Vec(min_count=20,
                     window=10,
                     size=200,
                     sample=5e-5,
                     alpha=0.01,
                     min_alpha=0.01,
                     # negative=20,
                     workers=4)

model.build_vocab(train_tokens, progress_per=10000)
model.train(train_tokens, total_examples=model.corpus_count, epochs=100, report_delay=1)
model.init_sims(replace=True)
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
