import numpy as np
from abc import ABC, abstractmethod


def logistic_cost(a, y):
    m = y.shape[1]
    cost = - (1./m) * np.sum(np.log(a) @ y.T + np.log(1 - a) @ (1 - y.T))
    return cost

def logistic_derivitave(a, y):
    grads = (-y/a) + ((1-y)/(1-a))
    return grads


class NetworkLayer(ABC):

    def __init__(self, size: int):
        self.size = size
        self.i = None
        self.w = None
        self.b = None
        self.z = None

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

        return self.activation(self.z)


    @abstractmethod
    def back_propagation(self, grads: np.matrix) -> np.matrix:
        dZ1 = (W2.T @ dZ2) * (1 - np.power(A1, 2))
        dW1 = (1. / m) * (self.z @ self.i.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)
        pass


class NeuralNetwork(object):

    def __init__(self, shape: list[NetworkLayer]) -> None:
        pass



