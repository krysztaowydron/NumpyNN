import numpy as np 

from keras.datasets import mnist

from NN import NN


# Preparing data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, (X_train.shape[0], 28*28))
X_train = X_train/255
X_test = np.reshape(X_test, (X_test.shape[0], 28*28))
X_test = X_test/255

new_y = np.zeros((len(y_train), 10)) # converting int to vector for example 1 -> [0,1,0,0,0,0,0,0,0,0]
for i in range(len(y_train)):
    new_y[i, y_train[i]] = 1
y_train = new_y

new_y = np.zeros((len(y_test), 10)) # converting int to vector for example 1 -> [0,1,0,0,0,0,0,0,0,0]
for i in range(len(y_test)):
    new_y[i, y_test[i]] = 1
y_test = new_y

net = NN((28*28, 100, 10))
net.info()
net.train(X_train, y_train, epoch=30, eta=0.5, lam=0.2, mini_batch_size=10,
            Test=True, X_test=X_test, y_test=y_test)


