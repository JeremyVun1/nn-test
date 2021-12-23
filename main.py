from random import randint
import matplotlib.pyplot as plt
import numpy as np

from data import TestData
from net import NeuralNetwork
from util import avg
from tests import test_nn

def main():
    nn = NeuralNetwork()
    test_nn(nn)

    avg_losses = []

    while True:
        training_data, test_data = get_data()

        predictions = nn.predict(test_data.data)
        losses = nn.calc_losses(predictions, test_data.labels)
        nn.train(training_data)

        avg_losses.append(np.average(losses))
        plot_graph(avg_losses)


def get_data(n = 5000):
    training_n = int(5000 * 0.7)
    test_n = n - training_n

    training_data = TestData()
    for _ in range(training_n):
        training_data.add(n, n*2)
    
    test_data = TestData()    
    for _ in range(test_n):
        n = randint(0, 10000)
        test_data.add(n, n*2)

    #base = [randint(0, 10000) for _ in range(test_n)]

    return training_data, test_data#, base


# plot a 2d graph
def plot_graph(*data):
    plt.clf()
    plt.rcParams["figure.figsize"] = (4, 2)
    plt.rcParams['toolbar'] = 'None'
    for d in data:
        plt.plot(d)
    plt.pause(0.001)


if __name__ == "__main__":
    main()