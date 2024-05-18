import tensorflow as tf


class DataLoader:
    def __init__(self, train_data, train_labels, train_unique_features):
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_unique_features = train_unique_features

    # TODO
    # def call(self, inputs):
