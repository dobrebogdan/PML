from nltk.tokenize import word_tokenize
import math
from nltk.stem.cistem import Cistem
import spacy
from sklearn import svm
nlp = spacy.load('de_core_news_sm')
stemmer = Cistem()

class Knn_classifier:

    def __init__(self, train_data, train_labels, validation_data, validation_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    # a. Write the classify_image(self, test_image, num_neighbors=3, metric='l2') function in order to classify 'test_image'
    # example using the k-NN method with 'num_neighbors' neighbors and 'metric' distance.

    def compute_distance(self, tweet1, tweet2):
        dist = 0.0
        l = len(tweet1)
        for i in range(0, l):
            dist = dist + (tweet1[i] - tweet2[i]) * (tweet1[i] - tweet2[i])
        return math.sqrt(dist)

    def classify_tweet(self, tweet):
        min_dist = 1000000000.0
        long = 52.0
        lat = 10
        l = len(self.train_data)
        for i in range(0,l):
            train_tweet = self.train_data[i]
            curr_disr = self.compute_distance(tweet, train_tweet)
            if curr_disr < min_dist:
                min_dist = curr_disr
                (long, lat) = self.train_labels[i]

        return str((long, lat))


    def classify_tweets(self):
        # write your code here
        clf = svm.SVC(kernel='rbf')
        clf.fit(self.train_data, self.train_labels)
        l = len(self.validation_data)
        self.predictions = clf.predict(self.validation_data)
        with open("output.txt", "w") as f:
            f.write(str(self.predictions))
        self.predictions = self.predictions
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