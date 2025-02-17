from typing import Callable
from numpy import array, exp, ndarray, dot


class ActivationFunctions:
    def sigmoid(self, x: float) -> float:
        """
        The sigmoid activation function is a smooth squashing function that
        squashes any unbounded input in the range [-infty, infty] to the range [0, 1].
        """
        return 1 / ( 1 + exp(-x) )


class Neuron:
    """
    A neuron is the basic unit of a neural network. It takes multiple inputs,
    multiples each by a weight, then adds a bias. The sum of the weighted inputs and bias is
    then passed through an activation function.

    The activation function turns unbounded inputs into bounded outputs.
    """
    def __init__(self, weights: ndarray, bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(self, input: ndarray, activation_function: Callable) -> float:
        """Feedforward mechanism - passing inputs forward through a neuron to get an output."""
        # weighted inputs, plus bias, then pass onto the activation function
        total: float = dot(self.weights, input) + self.bias
        return activation_function(total)


class NeuralNetwork:
    """
    A simple neural network with:
        - two inputs
        - one hidden layer with two neurons (h1, h2)
        - one output layer with one neuron (o1)
    Each neuron has the same weights and biases.
        - w = [0, 1]
        - b = 0
    """
    def __init__(self) -> None:
        weights: ndarray = array([0, 1])
        bias: float = 0
        # # Weights
        # self.w1: float = random.normal()
        # self.w2: float = random.normal()
        # self.w3: float = random.normal()
        # self.w4: float = random.normal()
        # self.w5: float = random.normal()
        # self.w6: float = random.normal()
        # weights: ndarray = array([self.w1, self.w2, self.w3, self.w4, self.w5, self.w6])
        # # Biases
        # self.b1: float = random.normal()
        # self.b2: float = random.normal()
        # self.b3: float = random.normal()
        # bias: ndarray = array([self.b1, self.b2, self.b3])

        self.h1: Neuron = Neuron(weights, bias)
        self.h2: Neuron = Neuron(weights, bias)
        self.o1: Neuron = Neuron(weights, bias)

    def feedforward(self, inputs: ndarray) -> float:
        out_h1: float = self.h1.feedforward(inputs, ActivationFunctions().sigmoid)
        out_h2: float = self.h2.feedforward(inputs, ActivationFunctions().sigmoid)
        out_o1: float = self.o1.feedforward(array([out_h1, out_h2]), ActivationFunctions().sigmoid)
        return out_o1


if __name__ == '__main__':
    weights: ndarray = array([0, 1])
    bias: float = 4
    inputs: ndarray = array([2, 3])

    nn: Neuron = Neuron(weights, bias)
    af: ActivationFunctions = ActivationFunctions()
    output: float = nn.feedforward(inputs, af.sigmoid)
    print(output)

    network: NeuralNetwork = NeuralNetwork()
    inputs: ndarray = array([2, 3])
    output: float = network.feedforward(inputs)
    print(output)

