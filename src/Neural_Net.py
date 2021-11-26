# 1d vector to represent the weights when using GA
# matrix to represent the weights when using ANN
import numpy as np
from types import FunctionType


n_input_neurons = 3
n_hidden_neurons = 4
n_output_neurons = 2

def Relu(inputs):
    output = np.maximum(0, inputs)
    return output

def Soft_max(inputs):
    exp_values = np.exp(inputs)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

X = np.array([3, 1, -1])
X2 = np.array([6, 6, 9, 9])

weights1 = np.array([[1, 2, 1],
                    [3, -2, 4],
                    [2, -1, -3],
                    [3, 3, 1],])
bias_1 = np.array([2, 3, 1, -2])

weights2 = np.array([[2, 1, 1, 2],
                    [4, 3, 1, -2],
                    ])
bias_2 = np.array([3, 1])


#input_l = [X1, X2]
weights_l = [weights1, weights2]
biases_l = [bias_1, bias_2]
activation_functions = [Relu, Relu]

print(weights2.shape[1])

class Layer:

    def __init__(self, weights: np.array, biases: np.array, activation_func: FunctionType ):
        if weights.shape[0] != len(biases):
            raise ValueError

        self.weights = weights
        self.biases = biases
        self.activation_func = activation_func
        
        weigths_copy = np.copy(weights)

    def forward(self, inputs):
        
        self.output = np.dot( self.weights, inputs) + self.biases
        return self.activation_func(self.output)

    def getMatrix(self):
        return self.weights

    def getBiasVector(self):
        return self.biases

    def getFunction(self):
        return self.activation_func


class NeuralNetwork:
    def __init__(self, weights_list, biases_list, functions):
        if len(weights_list) != len(biases_list) != len(functions):
            raise ValueError
        
        self.weights_list = weights_list
        self.biases_list = biases_list
        self.functions = functions
        self.layers = list(zip(weights_l, biases_l, activation_functions))

    def propagate(self, inputs):
        prev_outputs = inputs
        for weights, biases, activation_function in self.layers:
            layer_init = Layer(weights, biases, activation_function)
            layer_output = layer_init.forward(prev_outputs)
            
            if layer_output.shape[0] != weights.shape[0]:
                raise ValueError


            prev_outputs = layer_output
        return prev_outputs  

    def getLayers(self):
        return self.layers


#layer1 = Layer(weights1, bias_1, Relu)
#print(layer1.forward(X))

#layer2 = Layer(weights2, bias_2, Soft_max)
#print(layer2.forward(X))

Net1 = NeuralNetwork(weights_l, biases_l, activation_functions)
outputs = Net1.propagate(X)
print(outputs)

