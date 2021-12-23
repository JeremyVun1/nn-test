import numpy as np
import math

class NeuralNetwork:
    def __init__(self):
        self.weights = [
            np.random.rand(1, 10),
            np.random.rand(10, 1)
            #np.random.rand(10, 1)
        ]

    def train(self, data, labels):
        pass

    def predict(self, data):
        predictions = []
        for d in data:
            output = self.feed_forward(d)
            predictions.append(output[0][0])

        return predictions

    # activation(data * w)
    def feed_forward(self, data):
        for layer in self.weights:
            data = self.relu(np.dot(data, layer))

        return data

    def relu(self, x):
        return np.maximum(0, x)

    def calc_losses(self, actual, expected):
        losses = []

        for i in range(len(actual)):
            a = actual[i]
            e = expected[i]
            losses.append(abs(a - e))

        return losses
