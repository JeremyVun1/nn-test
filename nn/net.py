import numpy as np
import math

class NeuralNetwork:
    def init(self, shape):
        if len(shape) < 2:
            raise ValueError("Must have atleast 2 layers")

        self.weights = [np.random.rand(shape[0], shape[1])]
        for i in range(1, len(shape)-1):
            self.weights.append(np.random.rand(shape[i], shape[i+1]))

    def train(self, dataset):
        predictions = self.predict(dataset.data)
        losses = self.calc_losses(predictions, dataset.labels)

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
