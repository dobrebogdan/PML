class Knn_classifier:

    def __init__(self, train_data, train_labels, validation_data, validation_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    # a. Write the classify_image(self, test_image, num_neighbors=3, metric='l2') function in order to classify 'test_image'
    # example using the k-NN method with 'num_neighbors' neighbors and 'metric' distance.

    def compute_distance(self, tweet1, tweet2):
        return abs(len(tweet1) - len(tweet2))

    def classify_tweet(self, tweet, num_neighbors=5):
        distances = [(0, 0, 0)] * len(self.train_data)
        for i in range(0, len(self.train_data)):
            distances[i] = (self.compute_distance(tweet, self.train_data[i]), self.train_labels[i][0], self.train_labels[i][1])
        distances = sorted(distances)
        # print(distances)
        meanLong = 0.0
        meanLat = 0.0
        for i in range(0, num_neighbors):
            meanLong += float(distances[i][1])
            meanLat += float(distances[i][2])
        meanLong /= num_neighbors
        meanLat /= num_neighbors
        return meanLong, meanLat





    def classify_tweets(self):
        # write your code here
        predictions = [(0.0, 0.0)] * len(self.validation_data)
        for i in range(0, len(self.validation_data)):
            predictions[i] = self.classify_tweet(self.validation_data[i])
            if(i %10 ==  0):
                print("STEP " + str(i))
        print(predictions)



    # c. Define a function to compute the accurracy score given the predicted labels and the ground-truth labels.
    def accuracy_score(self):
        # write your code here
        pass