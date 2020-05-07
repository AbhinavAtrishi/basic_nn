import numpy as np


class InputLayer:
    def __init__(self, x, dimensions=(1,2)):
        self.size = dimensions
        self.upstream_gradient = None
        self.weights = np.random.random(dimensions)
        self.bias = 0
        self.x = x

    def forward(self):
        return (self.weights @ self.x) + self.bias

    def backprop(self, lr=1):
        if self.upstream_gradient is not None:
            dzdw = self.upstream_gradient @ self.x.T
            dzdb = np.sum(self.upstream_gradient)

            self.weights = self.weights - (lr * dzdw)
            self.bias = self.bias - (lr * dzdb)

            self.upstream_gradient = None


class Sigmoid:
    def __init__(self):
        self.previous_out = None
        self.upstream_gradient = None

    def forward(self, x):
        self.previous_out = 1/(1 + np.exp(-x))
        return self.previous_out

    def backprop(self, lr=1):
        return self.upstream_gradient * self.previous_out * (1 - self.previous_out)


class TerminalLayer:
    def __init__(self, y):
        self.expected_output = y

    def forward(self, predictions):
        return np.sum(np.power((self.expected_output - predictions), 2)) / (2 * len(self.expected_output))

    def backprop(self, predictions):
        return -(self.expected_output - predictions) / len(self.expected_output)
