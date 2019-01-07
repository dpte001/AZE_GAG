import numpy as np
from Convolution import im2col

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - self.pool_h) / self.stride)
        out_w = int(1+ (H + 2*self.pad - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최대값 (2)
        out = np.max(col, axis =1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return

data = np.arange(1*3*28*28).reshape(1,3,28,28)
example = Pooling(5, 5, 5)
example.forward(data)