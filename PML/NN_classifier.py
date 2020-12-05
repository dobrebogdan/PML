from nltk.tokenize import word_tokenize
import math
from nltk.stem.cistem import Cistem
import spacy
from sklearn import svm
nlp = spacy.load('de_core_news_sm')
stemmer = Cistem()

mode = "test"
# dev or test

class NN_classifier:

    def __init__(self, train_data, train_labels, validation_ids, validation_data, validation_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_ids = validation_ids
        self.validation_data = validation_data
        self.validation_labels = validation_labels


    def classify_tweets(self):
        # write your code here
        clf = svm.SVC(kernel='rbf')
        clf.fit(self.train_data, self.train_labels)
        l = len(self.validation_data)
        self.predictions = clf.predict(self.validation_data)
        with open("output.txt", "w+") as f:
            l = len(self.predictions)

            for i in range(0,l):
                f.write(self.validation_ids[i])
                f.write(", ")
                f.write(self.predictions[i])
                f.write("\n")
        self.predictions = self.predictions
        if mode == "dev":
            self.print_error()

    def get_touple(self, s):
        s = s.split(",")
        return (float(s[0].replace("(", "")), float(s[1].replace(")", "")))

    def print_error(self):
        error = 0.0
        for i in range(0, len(self.predictions)):
            (x1, y1) = self.get_touple(self.predictions[i])
            (x2, y2) = self.get_touple(self.validation_labels[i])
            curr_error = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            if curr_error < 0.000001:
                print("#####")
            error += curr_error
        error = error / len(self.predictions)
        print("ERROR IS")
        print(error)