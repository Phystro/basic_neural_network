# basic_neural_network
Basic Neural Network implementation form scratch

# Neural Networks

### Abstract

An artificial neural network is at its core, a mathematical equation.

The essence of an artificial neuron is the simple equation:

$$
f(x) = Z(x) = w \cdot x + b
$$

where; $w$ is the weight, $x$ is the input, $b$ is the bias and $Z(x)$ is the result. The AI system maps an input $x$ to a preferred output value $Z(x)$. $w$ and $b$ are determined through training. We have to train the parameters into the AI system.

#### Neurons

Neurons are the basic building blocks of a neural network. A neuron takes inputs, does computation with them, and produces one output. *(consider a curious case of getting as many outputs as inputs)*

Consider the case of a 2-input neuron;
- each input is multiplied by a weight:
	- $x_{1} \rightarrow x_{1} \times w_{1}$
	- $x_{2} \rightarrow x_{2} \times w_{2}$
- all weighted inputs are added together with a bias
	- $(x_{1} \times w_{1}) + (x_{2} \times w_{2}) + b$
	- $\sum\limits_{i}^{n} (x_{i}\cdot w_{i}) + b$
- the sum is passed through as activation function
	- $y = f(x_{1} \times w_{1} + x_{2} \times w_{2} + b)$
	- $y = f( \sum\limits_{i}^{n} (x_{i}\cdot w_{i}) + b )$

An *activation function* is used to turn an unbounded input into an output that has a nice predictable form. `normalisation` e.g. using the *sigmoid function*. Sigmoid function only outputs numbers in the range (0, 1) thus compressing any range $(-\infty, \infty)$ to (0, 1).

*Feedforward* - process of passing inputs forward to get an output.

#### Neural Network

A neural network is combination of several neurons connected together. A *hidden layer* is any layer between the input layer (first) and the output layer (last). There can be multiple hidden layers.

A network can have any number of layers and any number of neurons in those layers.

Example of a neural network:

Let all neurons have the same weight $[0, 1]$ and the same bias $b=0$, and the same sigmoid activation function. We denote $h_{1,}h_{2,}0_{1}$ as the outputs of the neurons they represent. Our inputs are $[2, 3]$:
$$
\begin{align*}
h_{1}=h_{2} &= f(w\cdot x + b)\\
&= f((0\cdot 2) + (1\cdot 3) + 0)\\
&= f(3)\\
&= 0.9526
\end{align*}
$$
$$
\begin{align*}
o_{1} &= f(w\cdot[h_{1},h_{2}] + b)\\
&= f((0\cdot h_{1}) + (1\cdot h_{2}) + b)\\
&= f(0.9526)\\
&= 0.7216
\end{align*}
$$


#### Loss

Loss is a way to quantify how 'well' trained our neural network is in order for us to make it 'better'.

*Training a network is all about minimising loss. The lower the loss, the better the predictions*

Using the **mean squared error** (MSE) loss:
$$
MSE = \frac{1}{n} \sum\limits_{i=1}^{n} (y_{true} - y_{pred})^2
$$
where;
- $n$ is the number of samples
- $y$ is the variable being predicted
- $y_{true}$ true value of the variable 'correct answer'
- $y_{pred}$ predicted value of the variable i.e. whatever our network outputs

$(y_{true} - y_{pred})^2$ is the **squared error**. Loss function simply takes the average over all squared errors.



