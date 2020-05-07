import numpy as np
from basic_nn import InputLayer, Sigmoid, TerminalLayer

x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([0, 1, 1, 1])
epochs = 1000


def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1 + np.exp(-x))


il = InputLayer(x)
sg = Sigmoid()
tl = TerminalLayer(y)

for itr in range(epochs):
    preds = sg.forward(il.forward())
    sg.upstream_gradient = tl.backprop(preds)
    il.upstream_gradient = sg.backprop()
    il.backprop()

print(f'Predictions : {preds}')
print(f'RMSE: {tl.forward(preds)}')