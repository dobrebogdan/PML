import matplotlib.pyplot as plt
import numpy as np
import csv
from Knn import Knn_classifier

# Press the green button in the gutter to run the script.
train_data = []
train_labels = []
train_data_dict = {}
with open("training.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        train_data.append(row[3])
        train_labels.append((row[1], row[2]))

        train_data_dict[(row[1], row[2])] = train_data_dict.get((row[1], row[2]), "") + " " + row[3]
print(len(train_data_dict))
print(len(train_data))

for pair in train_data_dict.items():
    print(pair)
    print("---------------------------------------------------")
validation_data = []
validation_labels = []
with open("validation.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        validation_data.append(row[3])
        validation_labels.append([row[1], row[2]])


train_data = np.array(train_data)
train_labels = np.array(train_labels)

validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)

knn_classifier = Knn_classifier(train_data, train_labels, validation_data, validation_labels)
#knn_classifier.classify_tweets()

# sunt mai multe tweeturi de la aceleasi persoane. Cate? Este un fenomen raspandit?
