from random import randint
from net import NeuralNetwork
import numpy as np

def test_nn():
    nn = NeuralNetwork()
    test_1(nn)
    test_2(nn)
    test_3(nn)
    print("Tests succesful!")
    input()

def test_1(nn):
    n = 10000
    data = [randint(0, 10000) * 2 for _ in range(n)]
    labels = data
    expected = [0 for _ in range(n)]

    actual = nn.calc_losses(data, labels)

    assert len(actual) == len(expected)
    assert np.array_equal(actual, expected)


def test_2(nn):
    data = [5, 4, 3, 2, 1]
    labels = [5, 4, 3, 2, 1]
    expected = [0, 0, 0, 0, 0]
    actual = nn.calc_losses(data, labels)
    
    assert len(actual) == len(expected)
    assert np.array_equal(actual, expected)

def test_3(nn):
    data = [5, 4, 3, 2, 1]
    labels = [0, 0, 0, 0, 0]
    expected = data
    actual = nn.calc_losses(data, labels)
    
    assert len(actual) == len(expected)
    assert np.array_equal(actual, expected)