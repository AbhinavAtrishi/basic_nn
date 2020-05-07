import numpy as np
from tqdm import tqdm

lr = 1
w = np.random.random((1, 2))
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([0, 1, 1, 1])
bias = 0
epochs = 1000


def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1 + np.exp(-x))


for itr in tqdm(range(epochs)):
    z_out = (w @ x) + bias
    sig_out = sigmoid(z_out)

    cost_b = -(y - sig_out) / len(y)
    sig_err = sigmoid(sig_out, True)
    sig_b = sig_err * cost_b

    dzdw = sig_b @ x.T
    dzdb = np.sum(sig_b)

    w = w - lr * dzdw
    bias = bias - lr * dzdb

print(f'Predictions : {sig_out}')
print(f'RMSE: {np.sum(np.power((y - sig_out), 2)) / (2 * len(y))}')