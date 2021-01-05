import csv
from SVM_classifier import SVM_classifier
import spacy
import gensim

# dev mode for calculating errors on validation data, test mode just for writing predictions to output.txt
mode = "dev"

# Loading the lemmatizer. It must first be downloaded
# python -m spacy download de_core_news_sm
nlp = spacy.load('de_core_news_sm')

# method that turns a text into a list of lemmatized words (tokens).
def text_to_tokens(curr_str):

    # removal of punctuation and other characters from the text
    for i in range(0, len(curr_str)):
        if (not curr_str[i] == " ") and (not curr_str[i].isalpha()):
            curr_str = curr_str.replace(curr_str[i], " ")

    # turining the text to lowercase
    curr_str = curr_str.lower()

    # Returns an object of type Doc, which is a sequence of Token objects
    tokens = nlp.tokenizer(curr_str)

    # A list of valid tokens
    good_tokens = []
    for token in tokens:
        # Gets the lemmatized token from the Token object
        token = str(token.lemma_)

        # Checks token validity
        if token == "" or (not token[0].isalpha()):
            continue

        # Adds token to list
        good_tokens.append(token)
    return good_tokens


""" Function that turns a text into a numerical array using its lemmatized tokens and the word2vec model.
    The first argument is the model, the second one is the text.
    For every word, the function computes the vector for its lemma, and then returns the average of the vectors
    as the vector for the text."""

def text_to_coords(model, curr_str):
    print("CONVERSION BEGIN")
    # get the lemmatized tokens from text
    tokens = text_to_tokens(curr_str)
    # the sum vector of all vectors of words that are found in the model's vocabulary
    sum = []
    cnt = 0.0
    for i in range(0, len(tokens)):
        if sum == []:
            # if the sum vector is empty, try to initialize it with the vector of the current word
            # if it exists in the model's vocabulary
            try:
                sum = model.wv[tokens[i]].copy()
                cnt += 1.0
            except:
                pass
        else:
            # try to add the vector for the current word to the sum of vectors if the world
            # exists in the model's vocabulary
            try:
                sum += model.wv[tokens[i]]
                cnt += 1.0
            except:
                pass

    # divide to the number of words that exist in the model's vocabulary to get the average
    sum = sum / cnt
    print("CONVERSION END")
    return sum


train_text = []
train_tokens = []
train_labels = []

# Load the train data into the train_text, train_tokens and train_labels
with open("training.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        train_text.append(row[3])
        train_tokens.append(text_to_tokens((row[3])))
        row[1] = float(row[1])
        row[2] = float(row[2])
        # the label will be a string form of the coordinates
        train_labels.append(str(row[1]) + "," + str(row[2]))
print("Train data loaded")

# Word2Vec is used to turn words into numerical vectors, which are then averaged to obtain a vector for a tweet
# Multiple tests were done and the parameters which behaved the best were selected
model = gensim.models.Word2Vec(min_count=20,
                     window=10,
                     size=300,
                     sample=5e-5,
                     alpha=0.01,
                     min_alpha=0.01,
                     workers=4)

# Building the model's vocabulary and training the model
model.build_vocab(train_tokens, progress_per=10000)
model.train(train_tokens, total_examples=model.corpus_count, epochs=100)

# Keeping only the current vectos to save memory
model.init_sims(replace=True)
print("Model created")

# Create a training data array of vector representations of sentences
train_data = []
for entry in train_text:
    train_data.append(text_to_coords(model, entry))


validation_data = []
validation_labels = []
validation_ids = []

# If the mode is dev, the test data will be loaded from the validation file, otherwise from the test file
validation_file = "test.txt"
if mode == "dev":
    validation_file = "validation.txt"

# Loading validation o test data
with open(validation_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        validation_ids.append(row[0])
        if mode == "test":
            # There are no labels in this case
            validation_data.append(text_to_coords(model, row[1]))
        else:
            # There are labels in this case
            validation_labels.append(str(row[1]) + "," + str(row[2]))
            validation_data.append(text_to_coords(model, row[3]))

# Use a SVM to classify validation / test data (numeric arrays) using training data
svm_classifier = SVM_classifier(train_data, train_labels, validation_ids, validation_data, validation_labels)

# Start classifying, it will write to output.txt in test case or print errors and write to output.txt in dev case
svm_classifier.classify_tweets()
