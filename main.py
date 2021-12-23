import numpy as np

from data import get_data
from net import NeuralNetwork
from tests import test_nn
from util import plot_graph

def main():
    nn = NeuralNetwork()

    avg_losses = []

    while True:
        training_data, test_data = get_data()

        predictions = nn.predict(test_data.data)
        losses = nn.calc_losses(predictions, test_data.labels)
        nn.train(training_data)

        avg_losses.append(np.average(losses))
        plot_graph(avg_losses)


if __name__ == "__main__":
    test_nn()
    main()