# Name: Kyle Rodrigues
# Date: 4/20/2021
# Lab 6

import torch
import numpy as np

# Layers in this file are arranged in roughly the order they
# would appear in a network.

# A layer in the network, could be an input, or a linear layer that takes multiple inputs
class Layer:
    def __init__(self, output_shape, train):
        """
        :param output_shape: the shape of the output
        """
        self.shape = output_shape
        self.output = np.zeros(output_shape)
        self.grad = torch.tensor(np.zeros(output_shape))
        self.train = train
    
    def clear_grad(self):
        """
        Clears the gradient
        """
        self.grad *= 0

    def accumulate_grad(self, gd):
        """
        :param gd: gradient to update self.grad with

        Updates the gradient
        """
        self.grad += gd


# The inputs for the neural network
class Input(Layer):
    def __init__(self, shape, train = False):
        """
        Sends the shape into the Layer class
        """
        Layer.__init__(self,shape, train)

    def set(self,output):
        """
        :param output: The output to set, as a numpy array. Raise an error if this output's size
                       would change.
        """
        if self.output.shape == output.shape:
            self.output = torch.tensor(output).double()
        else:
            assert("Expected output shape does not match inputed dimensions")

    def randomize(self):
        """
        Set the output of this input layer to random values sampled from the standard normal
        distribution (numpy has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.tensor(np.random.normal(0, 0.1, self.shape))

    def forward(self):
        """
        Nothing happens on forward pass with inputs
        """
        pass
    
    def backward(self):
        """
        Nothing happens on backwards pass with inputs
        """
        pass

    def step(self, alpha):
        """
        Update gradients with learning rate
        """
        if self.train == True:
            self.output -= alpha * self.grad


# Linear Regression
class Linear(Layer):
    def __init__(self, inputs, weights, bias):
        """
        :param x: the inputs
        :param w: the weights
        :param b: the bias
        """
        Layer.__init__(self, bias.shape, False)
        self.x = inputs
        self.w = weights
        self.b = bias

    def forward(self):
        """
        Apply forward pass: x*w + b
        """
        for a in range(self.shape):
            self.output[a] = torch.dot(self.x.output, self.w.output[a]) + self.b.output[a]
        self.output = torch.tensor(self.output)

    def backward(self):
        """
        Apply backward pass to calculate partial derivative with respect to x, w and b
        """
        djdw = self.grad[..., np.newaxis] @ self.x.output[...,np.newaxis].T
        self.w.accumulate_grad(djdw)

        djdx = self.grad @ self.w.output
        self.x.accumulate_grad(djdx)

        djdb = self.grad
        self.b.accumulate_grad(djdb)


    def step(self, alpha):
        """
        Update gradients with learning rate
        """
        self.w.step(alpha)
        self.x.step(alpha)
        self.b.step(alpha)


# ReLU activation function
class ReLU(Layer):
    def __init__(self, input_layer):
        """
        :param input_layer: the layer to apply ReLU to
        """
        Layer.__init__(self, input_layer.shape, False)
        self.input = input_layer

    def forward(self):
        """
        Apply ReLU to all output of the inputted layer
        """
        for a in range(len(self.output)):
            self.output[a] = self.input.output[a] * (self.input.output[a] > 0)
        self.output = torch.tensor(self.output)

    def backward(self):
        """
        Apply backward pass to calculate partial derivative with respect to output
        """
        djdz = self.grad * self.output
        self.input.accumulate_grad(djdz)

    def step(self, alpha):
        """
        Update gradients with learning rate
        """
        pass


# Gets the L2 loss of desired output and expected output
class Loss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the squared L2 norm of the inputs.
    """
    def __init__(self, input_value, y):
        """
        :param input_value: the predicted output
        :param y: the expected output
        """
        Layer.__init__(self, 1, False)
        self.input = input_value
        self.y = y

    def forward(self):
        """
        Get the loss after first applying softmax
        """
        # Softmax
        exp = torch.exp(self.input.output - torch.max(self.input.output))
        self.softmax = exp / exp.sum()

        # L2 Loss
        self.output = (1/2) * (torch.norm((self.softmax - self.y.output) ** 2))

    def backward(self):
        """
        Apply backward pass to calculate partial derivative with respect to input
        """
        djdl = self.grad * (self.softmax - self.y.output)
        self.input.accumulate_grad(djdl)

        djdy = self.grad * (self.softmax - self.y.output)
        self.y.accumulate_grad(djdy)

    def step(self, alpha):
        """
        Update gradients with learning rate
        """
        pass


# Sums 2 values
class Sum(Layer):
    def __init__(self, val1, val2):
        """
        :param val1: the first value to sum
        :param val2: the second value to sum
        """
        Layer.__init__(self, 1, False)
        self.val1 = val1
        self.val2 = val2

    def forward(self):
        """
        Add val1 to val2
        """
        self.output = self.val1.output + self.val2.output

    def backward(self):
        """
        Apply backward pass to calculate partial derivative with respect to input values
        """
        djd1 = self.grad
        self.val1.accumulate_grad(djd1)

        djd2 = self.grad
        self.val2.accumulate_grad(djd2)

    def step(self, alpha):
        """
        Update gradients with learning rate, do I pass here?
        """
        pass


# Regularization
class Regularization(Layer):
    def __init__(self, val1, beta):
        """
        :param val1: the first value to sum
        :param val2: the second value to sum
        """
        Layer.__init__(self, 1, False)
        self.val1 = val1
        self.beta = beta

    def forward(self):
        """
        Regularize
        """
        self.output = (self.beta/2) * (torch.norm(self.val1.output) ** 2)

    def backward(self):
        """
        Apply backward pass to calculate partial derivative with respect to input values
        """
        djd1 = self.grad * self.beta * self.val1.output
        self.val1.accumulate_grad(djd1)

    def step(self, alpha):
        """
        Update gradients with learning rate, do I pass here?
        """
        pass

