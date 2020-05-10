import numpy as np


class InputLayer:
    def __init__(self, x, dimensions=(1,2), activation='sigmoid'):
        self.size = dimensions
        self.upstream_gradient = None
        self.weights = np.random.random(dimensions)
        self.bias = 0
        self.x = x
        self.activation = activation
        self.output = None

    def __activate(self, x, deriv=False):
        if self.activation == 'sigmoid':
            if deriv:
                return x * (1 - x)
            return 1/(1 + np.exp(-x))
        elif self.activation == 'relu':
            if deriv:
                return 1 * (x > 0)
            return x * (x > 0)
        elif self.activation == 'linear':
            if deriv:
                return np.ones(x.shape)
            return x
        else:
            raise NotImplementedError

    def forward(self, x):
        self.output = self.__activate((self.weights @ x) + self.bias)
        return self.output

    def backprop(self, lr=1):
        if self.upstream_gradient is not None:
            activation_upstream = self.upstream_gradient * self.__activate(self.output, True)
            dzdw = activation_upstream @ self.x.T
            dzdb = np.sum(activation_upstream)

            self.weights = self.weights - (lr * dzdw)
            self.bias = self.bias - (lr * dzdb)

            self.upstream_gradient = None


class TerminalLayer:
    def __init__(self, y):
        self.expected_output = y

    def forward(self, predictions):
        return np.sum(np.power((self.expected_output - predictions), 2)) / (2 * len(self.expected_output))

    def backprop(self, predictions):
        return -(self.expected_output - predictions) / len(self.expected_output)
