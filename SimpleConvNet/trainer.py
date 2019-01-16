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
        self.current_iter =  0
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

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        print("train loss = " + str(loss))
        self.current_iter += 1

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train, t_train = self.x_train, self.t_train
            x_test, t_test = self.x_test, self.t_test

            train_acc = self.network.accuracy(x_train, t_train)
            test_acc = self.network.accuracy(x_test, t_test)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append((test_acc))
            print("===epochs = " + str(self.current_epoch) + ", train acc = " + str(train_acc) + ", test acc = " +str(test_acc) + "===")


    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        print('=================Final Test Accuracy===================')
        print(test_acc)