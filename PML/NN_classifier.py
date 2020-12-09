from nltk.tokenize import word_tokenize
import math
from nltk.stem.cistem import Cistem
from model import NeuralNet
import spacy
import random
import json
import torch
nlp = spacy.load('de_core_news_sm')
stemmer = Cistem()

mode = "dev"
# dev or test

class NN_classifier:

    def __init__(self, train_data, train_labels, validation_ids, validation_data, validation_labels, labels_dict):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_ids = validation_ids
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.labels_dict = labels_dict


    def classify_tweets(self):
        # write your code here
        file = "data.pth"
        data = torch.load(file)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        state_dict = data["model_state"]
        model = NeuralNet(input_size, hidden_size, output_size)
        model.eval()

        l = len(self.validation_data)
        self.predictions = [""] * l
        for i in range(0,l):
            curr_data = self.validation_data[i]
            curr_data = curr_data.reshape(1, curr_data.shape[0])
            curr_data = torch.from_numpy(curr_data)
            output = model(curr_data)
            _, predicted = torch.max(output, dim=1)
            tag = predicted.item()
            self.predictions[i] = self.train_labels[tag]
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            print("#")
            print(prob)
            # maybe use prob for another idea

        with open("output.txt", "w+") as f:
            l = len(self.predictions)

            for i in range(0,l):
                f.write(self.validation_ids[i])
                f.write(", ")
                f.write(self.labels_dict[self.predictions[i]])
                f.write("\n")
        if mode == "dev":
            self.print_error()

    def get_touple(self, s):
        s = s.split(",")
        return (float(s[0].replace("(", "")), float(s[1].replace(")", "")))

    def print_error(self):
        error = 0.0
        for i in range(0, len(self.predictions)):
            (x1, y1) = self.get_touple(self.labels_dict[self.predictions[i]])
            (x2, y2) = self.get_touple(self.labels_dict[self.validation_labels[i]])
            curr_error = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            if curr_error < 0.000001:
                print("#####")
            error += curr_error
        error = error / len(self.predictions)
        print("ERROR IS")
        print(error)