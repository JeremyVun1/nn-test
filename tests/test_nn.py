from random import randint
import numpy as np

from nn.net import NeuralNetwork
from unittest import TestCase
import unittest

class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()
        
    def test_calc_loss_1(self):
        n = 10000
        data = [randint(0, 10000) * 2 for _ in range(n)]
        labels = data
        expected = [0 for _ in range(n)]

        actual = self.nn.calc_losses(data, labels)

        assert len(actual) == len(expected)
        assert np.array_equal(actual, expected)


    def test_calc_loss_2(self):
        data = [5, 4, 3, 2, 1]
        labels = [5, 4, 3, 2, 1]
        expected = [0, 0, 0, 0, 0]
        actual = self.nn.calc_losses(data, labels)
        
        assert len(actual) == len(expected)
        assert np.array_equal(actual, expected)

    def test_calc_loss_3(self):
        data = [5, 4, 3, 2, 1]
        labels = [0, 0, 0, 0, 0]
        expected = data
        actual = self.nn.calc_losses(data, labels)
        
        assert len(actual) == len(expected)
        assert np.array_equal(actual, expected)

    def test_init_weights(self):
        shape = [1, 10, 1]
        self.nn.init(shape)

        assert len(self.nn.weights) == len(shape) - 1

    def test_init_weights(self):
        for _ in range(randint(1, 10)):
            shape = [randint(1, 10) for _ in range(randint(2, 10))]
            self.nn.init(shape)
            assert len(self.nn.weights) == len(shape) - 1

        shape = [1]
        with self.assertRaises(ValueError):
            self.nn.init(shape)