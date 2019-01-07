import numpy as np

def softmax(x):
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        C = np.max(x[0])
        exp_x = np.exp(x[0][:]-C)
        y[i] = exp_x / np.sum(exp_x)
    return y
