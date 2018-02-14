import numpy as np
from abc import ABC, abstractmethod

class NetworkLayer(ABC):

    def __init__(self, size: int):
        self.size = size
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

        self.z = (self.w @ inputs) + self.b

        return self.activation(self.z)


    @abstractmethod
    def back_propagation(self, grads: np.matrix) -> np.matrix:
        pass


class NeuralNetwork(object):

    def __init__(self, shape: list[NetworkLayer]) -> None:
        pass



