import numpy as np

from .net import NeuralNetwork
from .data import get_data
from .util import plot_graph

def train():
    nn = NeuralNetwork()
    shape = [1, 10, 1]
    nn.init(shape)

    avg_losses = []

    while True:
        training_data, test_data = get_data()

        predictions = nn.predict(test_data.data)
        losses = nn.calc_losses(predictions, test_data.labels)
        nn.train(training_data)

        avg_losses.append(np.average(losses))
        plot_graph(avg_losses)
