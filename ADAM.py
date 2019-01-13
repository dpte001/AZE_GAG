import numpy as np

class Adam:
    def __init__(self, lr = 0.001, beta1 =0.9, beta2 = 0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m == None:
            self.m, self.n = {}, {}
            for key, val in params.item():
                m[key] = np.zeros_like(val)
                v[key] = np.zeros_like(val)

        self.iter += 1
        lr._t = lr*np.sqrt(1-self.beta2**self.iter)/(1-self.beta1**self.iter)

        for key in params:
            self.m[key] += (1-self,beta1) * (self.grads[key] - self.m[key])
            self.v[key] += (1-self.beta2) * (self.grads[key]**2 - self.v[key])
            params[key] -= lr_t*self.m[key]/np.sqrt(self.v + 1e-7)



