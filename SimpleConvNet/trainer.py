from ADAM import *
import numpy as np

class trainer:
    def __init__(self, network, x_train, x_test, t_train, t_test,
                 epochs =20, mini_batch_size = 100,
                optimizer_param = {'lr' : 0.01,} ):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.optimizer = Adam(**optimizer_param)
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch,t_batch)
        self.optimizer.update(self.network.params, grads)

    def train(self):
        for i in range(self.max_iter):
            self.train_step()
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        print('=================Final Test Accuracy===================')
        print(test_acc)