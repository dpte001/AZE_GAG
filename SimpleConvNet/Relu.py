import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


data = np.arange(1 * 3 * 4 * 4).reshape(1, 3, 4, 4) - 24
dout = np.ones((1,3,4,4))
example = Relu()
example.forward(data)

print(example.backward(dout))