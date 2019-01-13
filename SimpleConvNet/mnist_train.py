from mnist import *
import SimpleConvNet
from trainer import *


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)

train_loss_list = []
x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(10000, 1, 28, 28)


#하이퍼파라미터

liters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_late = 0.1

network = SimpleConvNet.SimpleConvNet()


example = trainer(network, x_train, x_test, t_train, t_test)

example.train()

