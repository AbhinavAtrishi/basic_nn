import numpy as np
from basic_nn import InputLayer, TerminalLayer

# Optional: Set a seed for reproducibility
np.random.seed(1)

# Current implementation supports 1 layer neural networks. These can learn simple linear separations
# Here we are implementing a simple AND gate
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([0, 1, 10, 11])
# x = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
# y = np.array([0, 11, 22, 33, 44])
epochs = 1000

# Input Layer
il = InputLayer(x, (1, 2), 'relu')
# Terminal Layer used for computing error & back propagation
tl = TerminalLayer(y)

for itr in range(epochs):
    preds = il.forward(x)
    # Set the upstream gradient for the previous layer
    il.upstream_gradient = tl.backprop(preds)
    il.backprop()

print(f'Predictions : {preds}')
print(f'RMSE: {tl.forward(preds)}')

x_unseen = x = np.array([[2, 1, 3, 5], [0, 1, 0, 1]])
pred_u = il.forward(x_unseen)
print("Unknown Dataset :", np.around(pred_u))
