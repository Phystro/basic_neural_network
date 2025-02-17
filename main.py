from typing import Callable
from numpy import append, apply_along_axis, array, exp, ndarray, dot, random
import matplotlib.pyplot as plt


class ActivationFunctions:
    def sigmoid(self, x: float) -> float:
        """
        The sigmoid activation function is a smooth squashing function that
        squashes any unbounded input in the range [-infty, infty] to the range [0, 1].
        """
        return 1 / ( 1 + exp(-x) )

    def deriv_sigmoid(self, x: float) -> float:
        """Derivative of the sigmoid function"""
        fx: float = self.sigmoid(x)
        return fx * (1 - fx)


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
        # weights: ndarray = array([0, 1])
        # bias: float = 0
        # Weights
        self.w1: float = random.normal()
        self.w2: float = random.normal()
        self.w3: float = random.normal()
        self.w4: float = random.normal()
        self.w5: float = random.normal()
        self.w6: float = random.normal()
        weights: ndarray = array([self.w1, self.w2, self.w3, self.w4, self.w5, self.w6])
        # Biases
        self.b1: float = random.normal()
        self.b2: float = random.normal()
        self.b3: float = random.normal()
        bias: ndarray = array([self.b1, self.b2, self.b3])

        # self.h1: Neuron = Neuron(weights, bias)
        # self.h2: Neuron = Neuron(weights, bias)
        # self.o1: Neuron = Neuron(weights, bias)

    def feedforward(self, inputs: ndarray) -> float:
        # out_h1: float = self.h1.feedforward(inputs, ActivationFunctions().sigmoid)
        # out_h2: float = self.h2.feedforward(inputs, ActivationFunctions().sigmoid)
        # out_o1: float = self.o1.feedforward(array([out_h1, out_h2]), ActivationFunctions().sigmoid)
        # return out_o1
        sigmoid: Callable = ActivationFunctions().sigmoid

        self.sum_h1: float = self.w1 * inputs[0] + self.w2 * inputs[1] + self.b1
        self.h1: float = sigmoid(self.sum_h1)

        self.sum_h2: float = self.w3 * inputs[0] + self.w4 * inputs[1] + self.b2
        self.h2: float = sigmoid(self.sum_h2)

        self.sum_o1: float = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
        self.o1: float = sigmoid(self.sum_o1)

        return self.o1

    def mse_loss(self, y_true: ndarray, y_pred: ndarray) -> float:
        # y_true and y_pred are numpy arrays of the same length.
        return ((y_true - y_pred) ** 2).mean()


    def train(self, data:  ndarray, all_y_trues: ndarray) -> None:
        """
        data is a(nx2) array, where n is the number of samples in the dataset
        all_y_trues has n elements
        elements in all_y_trues correspond to those in the data
        """
        learn_rate: float = 0.1
        epochs: int = 1000 # number of times to loop through the entire dataset

        epochs_data: ndarray = array([])
        loss_data: ndarray = array([])

        for epoch in range(epochs):
            for inputs, y_true in zip(data, all_y_trues):
                # forward feed
                y_pred: float = self.feedforward(inputs)

                # Calculate the partial derivatives
                deriv_sigmoid: Callable = ActivationFunctions().deriv_sigmoid

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = self.h1 * deriv_sigmoid(self.sum_o1)
                d_ypred_d_w6 = self.h2 * deriv_sigmoid(self.sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(self.sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(self.sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(self.sum_o1)

                # Neuron h1
                d_h1_d_w1 = inputs[0] * deriv_sigmoid(self.sum_h1)
                d_h1_d_w2 = inputs[1] * deriv_sigmoid(self.sum_h1)
                d_h1_d_b1 = deriv_sigmoid(self.sum_h1)

                # Neuron h2
                d_h2_d_w3 = inputs[0] * deriv_sigmoid(self.sum_h2)
                d_h2_d_w4 = inputs[1] * deriv_sigmoid(self.sum_h2)
                d_h2_d_b2 = deriv_sigmoid(self.sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds: ndarray = apply_along_axis(self.feedforward, 1, data)
                    loss: float = self.mse_loss(all_y_trues, y_preds)
                    print(f'Epoch: {epoch}, Loss: {loss}')
                    epochs_data = append(epochs_data, epoch)
                    loss_data = append(loss_data, loss)

        self.plot_loss(epochs_data, loss_data)


    def plot_loss(self, epochs_data: ndarray, loss_data: ndarray) -> None:
        """Make a plot of the neural network loss vs. epochs"""
        plt.plot(epochs_data, loss_data)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Neural Network Loss vs. Epochs')
        plt.show()


if __name__ == '__main__':
    weights: ndarray = array([0, 1])
    bias: float = 4
    inputs: ndarray = array([2, 3])

    nn: Neuron = Neuron(weights, bias)
    af: ActivationFunctions = ActivationFunctions()
    output: float = nn.feedforward(inputs, af.sigmoid)
    print(f'One neuron: {output}')

    network: NeuralNetwork = NeuralNetwork()
    inputs: ndarray = array([2, 3])
    output: float = network.feedforward(inputs)
    print(f'network: {output}')

    # dataset
    data: ndarray = array([
        [-2, -1], # Alice
        [25, 6], # Bob
        [17, 4], # Charlie
        [-15, -6], # Jane
        ])

    all_y_trues: ndarray = array([
        1, # Alice
        0, # Bob
        0, # Charlie
        1 # Jane
        ])

    # Train neural network
    network: NeuralNetwork = NeuralNetwork()
    network.train(data, all_y_trues)

    # Make some predictions, given weight and height of people, if they are male or female
    emily: ndarray = array([-7, -3])
    frank: ndarray = array([20, 2])
    print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M

