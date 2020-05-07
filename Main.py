import numpy as np
from basic_nn import InputLayer, Sigmoid, TerminalLayer

x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([0, 0, 0, 1])
epochs = 1000

il = InputLayer(x)
tl = TerminalLayer(y)

for itr in range(epochs):
    preds = il.forward()
    il.upstream_gradient = tl.backprop(preds)
    il.backprop()

print(f'Predictions : {preds}')
print(f'RMSE: {tl.forward(preds)}')