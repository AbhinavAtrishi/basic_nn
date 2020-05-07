import numpy as np
from basic_nn import InputLayer, TerminalLayer

# Optional: Set a seed for reproducibility
np.random.seed(1)

# Current implementation supports 1 layer neural networks. These can learn simple linear separations
# Here we are implementing a simple AND gate
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([0, 0, 0, 1])
epochs = 1000

# Input Layer
il = InputLayer(x, (1, 2), 'sigmoid')
# Terminal Layer used for computing error & back propagation
tl = TerminalLayer(y)

for itr in range(epochs):
    preds = il.forward()
    # Set the upstream gradient for the previous layer
    il.upstream_gradient = tl.backprop(preds)
    il.backprop()

print(f'Predictions : {preds}')
print(f'RMSE: {tl.forward(preds)}')
