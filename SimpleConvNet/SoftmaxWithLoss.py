import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    y = exp_x.copy()
    for i in range(x.shape):
        sum_exp_x = np.sum(exp_x[i])
        y[i] = y[i] / sum_exp_x
    return y

# 교차 엔트로피 (one-hot inocoding)
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = np.reshape(1, t.size)
        y = np.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size


class SoftmaxWithLoss
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def foward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
