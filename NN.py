import numpy as np

"""
Nerual network using cross entropy as a cost function with L2 regulization to prevent overfitting
"""

class NN:
    def __init__(self, shape):
        self.shape = shape
        self.layers = len(shape)
        self.weights = np.array([np.random.randn(shape[i-1], shape[i]) for i in range(1, self.layers)])
        self.biases = np.array([np.random.randn(shape[i]) for i in range(1, self.layers)])

    def info(self):
        print(f'Shape: {self.shape}')
        print(f'Layers: {self.layers}')
        print('Weights:')
        for x in self.weights:
            print(x.shape)
        print('Biases:')
        for x in self.biases:
            print(x.shape)

    def feed(self, X):
        output = X
        for i in range(self.layers-1):
            output = sigmoid_function(np.matmul(output, self.weights[i]) + self.biases[i])

        return output

    def train(self, X_train, y_train, epoch, eta, lam, mini_batch_size, Test=False, X_test=None, y_test=None):
        for e in range(epoch):
            print(f'Epoch {e+1}:')
            ## Spliting train set into mini batches
            train_data = [X_train, y_train]
            n = X_train.shape[0]
            mini_batches = [[train_data[0][k:k+mini_batch_size], train_data[1][k:k+mini_batch_size]] 
                for k in range(int(X_train.shape[0]/mini_batch_size))]
            
            for mini_batch in mini_batches:
                self.gradien_descend(mini_batch, eta, lam, n)

            ## Test results
            if Test:
                output = self.predict(X_test)
                new_y = np.argmax(y_test, axis=1)
                cnt = 0
                for o, y in zip(output, new_y):
                    if o == y:
                        cnt += 1
                print(f'Accuracy: {cnt/len(y_test)}')
            else:
                print('No test data')

    def gradien_descend(self, mini_batch, eta, lam, n):
        X = mini_batch[0]
        new_y = mini_batch[1] # y in vector for example [0,1,0,0,0,0,0,0,0,0] 
        mini_batch_size = len(new_y)
        
        
        weights_gradient = np.zeros_like(self.weights)
        biases_gradient = np.zeros_like(self.biases)


        for x, y in zip(X, new_y):
            wg, bg = self.back_propagation(x, y)
            weights_gradient += wg
            biases_gradient += bg

        for i in range(self.layers-1):
            self.weights[i] = ((1 - (eta*lam/n))*self.weights[i]) - ((eta/mini_batch_size)*weights_gradient[i])
            self.biases[i] = self.biases[i] - ((eta/mini_batch_size)*biases_gradient[i])

    def back_propagation(self, x, y):
        zs = [] #array to store z=w*input +b
        output = x
        outputs = [x] #output=sigmoid(z)

        weights_gradient = np.zeros_like(self.weights)
        biases_gradient = np.zeros_like(self.biases)

        # forward pass
        for i in range(self.layers-1):
            z = np.matmul(output, self.weights[i]) + self.biases[i]
            zs.append(z)
            output = sigmoid_function(z)
            outputs.append(output)
        
        delta = cost_derivative(outputs[-1], y)

        biases_gradient[-1] = delta
        weights_gradient[-1] = np.outer(np.transpose(outputs[-2]), delta)

        # backward pass
        for i in range(self.layers-2, 0, -1):
            delta = np.dot(self.weights[i], delta) * sigmoid_derivative(zs[i-1])
            biases_gradient[i-1] = delta
            weights_gradient[i-1] = np.outer(outputs[i-1], delta)

        return weights_gradient, biases_gradient
        
    def predict(self, X):
        output = self.feed(X)
        return np.argmax(output, axis=1)

def sigmoid_function(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid_function(z)*(1- sigmoid_function(z))


def cost_derivative(predict, true): #derivative of cost funcion in term of predict dC/dpredict
    return predict - true