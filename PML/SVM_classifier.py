import math
import spacy
from sklearn import svm

# Loading the lemmatizer. It must first be downloaded
nlp = spacy.load('de_core_news_sm')

mode = "dev"
# dev mode for calculating errors on validation data, test mode for writing predictions to output.txt

class SVM_classifier:

    def __init__(self, train_data, train_labels, validation_ids, validation_data, validation_labels):
        # Get the data
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_ids = validation_ids
        self.validation_data = validation_data
        self.validation_labels = validation_labels


    def classify_tweets(self):
        # Using SVC with a rbf kernel because it yielded the best results at tests
        clf = svm.SVC(kernel='rbf')
        print("Before fitting")
        # Fitting the training data
        clf.fit(self.train_data, self.train_labels)
        print("After fitting")
        print("Before prediction")
        # Predicting the labels of the validation / test data
        self.predictions = clf.predict(self.validation_data)
        print("After prediction")
        # Writing predictions to output.txt
        with open("output.txt", "w+") as f:
            l = len(self.predictions)
            for i in range(0,l):
                f.write(self.validation_ids[i])
                f.write(", ")
                f.write(self.predictions[i])
                f.write("\n")
        print("After file writing")
        if mode == "dev":
            # in dev we have the validation labels, so we can print errors
            self.print_error()

    # turns the coordinates from string form to touple form
    def get_touple(self, s):
        s = s.split(",")
        return (float(s[0].replace("(", "")), float(s[1].replace(")", "")))

    # prints the errors of the prediction in the dev case
    def print_error(self):
        abs_error = 0.0
        squared_error = 0.0
        for i in range(0, len(self.predictions)):
            (x1, y1) = self.get_touple(self.predictions[i])
            (x2, y2) = self.get_touple(self.validation_labels[i])
            curr_error = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            # add current error to absolute error and mean squared error
            abs_error += curr_error
            squared_error += curr_error * curr_error

        # get the average errors
        mean_abs_error = abs_error / len(self.predictions)
        mean_squared_error = squared_error / len(self.predictions)
        print("MEAN ABSOLUTE ERROR IS:")
        print(mean_abs_error)
        print("MEAN SQUARED ERROR is:")
        print(mean_squared_error)
