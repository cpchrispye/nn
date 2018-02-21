import numpy as np
from abc import ABC, abstractmethod
from typing import List


def logistic_cost(a, y):
    m = y.shape[1]
    errors = np.log(a) @ y.T + np.log(1 - a) @ (1 - y.T)
    cost = - (1./m) * np.sum(errors)
    return cost

def logistic_derivitave(a, y):
    grads = (-y/a) + ((1-y)/(1-a))
    return np.nan_to_num(grads)


class NetworkLayer(ABC):

    def __init__(self, size: int):
        self.size = size

        self.i = None
        self.w = None
        self.b = None
        self.z = None
        self.a = None

    @abstractmethod
    def activation(self, z: np.matrix)-> np.matrix:
        pass

    @abstractmethod
    def activation_derivative(self, z: np.matrix)-> np.matrix:
        pass

    def forward_propagation(self, inputs: np.matrix) -> np.matrix:
        if self.w is None:
            frm  = inputs.shape[0]
            to   = self.size
            self.w = np.random.randn(to, frm) * 0.01
            self.b = np.zeros((to, 1))

        self.i = inputs
        self.z = (self.w @ inputs) + self.b
        a = self.activation(self.z)

        return a


    def back_propagation(self, da: np.matrix, alpha=0.01) -> np.matrix:

        dZ = np.multiply(da, self.activation_derivative(self.z))
        m = da.shape[1]
        dW = (1. / m) * (dZ @ self.i.T)
        dB = (1. / m) * np.sum(np.array(dZ), axis=1, keepdims=True)

        da_out = self.w.T @ dZ

        self.w -= alpha * dW
        self.b -= alpha * dB
        # da_out = None
        return da_out

class TanhLayer(NetworkLayer):

    def activation(self, z: np.matrix) -> np.matrix:
        return np.tanh(z)

    def activation_derivative(self, z: np.matrix) -> np.matrix:
        return (1 - np.power(z, 2))


class SigmoidLayer(NetworkLayer):

    def activation(self, z: np.matrix) -> np.matrix:
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z: np.matrix) -> np.matrix:
        a = self.activation(z)
        return np.multiply(a, (1 - a))

class SinLayer(NetworkLayer):

    def activation(self, z: np.matrix) -> np.matrix:
        return np.sin(z)

    def activation_derivative(self, z: np.matrix) -> np.matrix:
        return np.cos(z)


class NeuralNetwork(object):

    def __init__(self, *layers: List[NetworkLayer]) -> None:
        self.layers = layers # type: List[NetworkLayer]

    def forward_propagation(self, inputs: np.matrix) -> np.matrix:
        a = inputs
        for layer in self.layers:
            a = layer.forward_propagation(a)
        return a

    def back_propagation(self, da: np.matrix, alpha=0.1) -> np.matrix:
        for layer in self.layers[::-1]:
            da = layer.back_propagation(da, alpha)
        return da

    def train(self, x, y, iter=1000, alpha=0.1):

        for i in range(iter):
            a = self.forward_propagation(x)

            cost = logistic_cost(a, y)
            print(i, cost)
            da = logistic_derivitave(a, y)

            self.back_propagation(da, min(1, cost * alpha))

        return np.sum(np.abs(np.round(self.forward_propagation(x)) - y)) / y.shape[1]


if __name__ == '__main__':
    nn = NeuralNetwork(
                        SinLayer(30),
                        TanhLayer(20),
                        SigmoidLayer(1)
                       )

    x = np.matrix([[0.1, 0.2, 0.3],
                   [0.2, 0.4, 0.5]])

    y = np.matrix([0, 0, 1])

    print(nn.train(x, y, 10000, 1))





