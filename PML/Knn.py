class Knn_classifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    # a. Write the classify_image(self, test_image, num_neighbors=3, metric='l2') function in order to classify 'test_image'
    # example using the k-NN method with 'num_neighbors' neighbors and 'metric' distance.
    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        # write your code here
        pass

    # b. Write the classify_images(self, test_images, num_neighbors=3, metric='l2') function in order to predict the labels of
    # the test images.
    def classify_images(self):
        # write your code here
        pass

    # c. Define a function to compute the accurracy score given the predicted labels and the ground-truth labels.
    def accuracy_score(self):
        # write your code here
        pass