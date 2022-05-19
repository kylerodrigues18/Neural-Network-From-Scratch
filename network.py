# Name: Kyle Rodrigues
# Date: 4/20/2021
# Lab 6

class Network:
    def __init__(self):
        """
        Initializes empty layers that holds the layers in the network
        """
        self.layers = []

    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        self.layers.append(layer)


    def forward(self):
        """
        Compute the output of the network in the forward direction.

        :param input: A numpy array that will serve as the input for this forward pass
        :return: A numpy array with useful output (e.g., the softmax decisions)
        """
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.
        for l in self.layers:
            l.forward()


    def backward(self):
        """
        Compute the output of the network in the backward direction.

        :param input: A numpy array that will serve as the input for this forward pass
        :return: A numpy array with useful output (e.g., the softmax decisions)
        """
        # Zero out gradients
        for l in self.layers:
            l.clear_grad()

        # Set last layers grad to be 1
        self.layers[-1].grad += 1

        # Compute gradients backward
        for l in range(len(self.layers), 0, -1):
            self.layers[l - 1].backward()


    def step(self, alpha):
        """
        Update gradients if needed.

        :param input: A numpy array that will serve as the input for this forward pass
        :return: A numpy array with useful output (e.g., the softmax decisions)
        """
        for l in self.layers:
            l.step(alpha)
        
        