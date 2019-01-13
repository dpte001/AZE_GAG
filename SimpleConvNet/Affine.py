import numpy as np
from Convolution import *
from Pooling import *
from Relu import *
from SoftmaxWithLoss import *


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0],-1)
        self.x = x
        print(self.x.shape)
        out = np.dot(self.x, self.W) + self.b
        print(out.shape)
        return out

    def backward(self, dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(self.original_x_shape)

        return dx



