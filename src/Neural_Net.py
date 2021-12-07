
import numpy as np
from types import FunctionType
import random
from numpy.core.fromnumeric import argmax
from numpy.lib.utils import safe_eval
import itertools


#activation functions
def Relu(inputs):
    output = np.maximum(0, inputs)
    return output


def Soft_max(inputs):
    exp_values = np.exp(inputs)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


# function to create a random lists of shapes and biasses based on the shape of the network
def init_weights_biases(net_shape, activation_func=None):
    weights = []
    biases = []
    for i,j in zip(net_shape,net_shape[1:]):
        weight = 0.10 * np.random.rand(j, i)
        bias = 0.20 * np.random.rand(j, 1)
        weights.append(weight)
        biases.append(bias)
    
    return weights, biases


# class to calculate a layer of a neural netork
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
        self.layers = list(zip(weights_list, biases_list, functions))

    def propagate(self, inputs):
        prev_outputs = inputs
        for weights, biases, activation_function in self.layers:
            layer_init = Layer(weights, biases, activation_function)
            layer_output = layer_init.forward(prev_outputs)
            if layer_output.shape[0] != weights.shape[0]:
                raise ValueError

            #print(layer_output)
            prev_outputs = layer_output
        return prev_outputs  

    def getLayers(self):
        return self.layers


### class player to play the game
class Zak_Player():
       
    def __init__(self, weights_list, biases_list, functions):
        self.weights_list = weights_list
        self.biases_list = biases_list
        self.functions = functions #[Relu, Relu]
        self.fitness = None
        self.NNet = NeuralNetwork(self.weights_list, self.biases_list, self.functions) # get all the combination of the moves
        self.possible_moves = [p for p in itertools.product([0, 1, 2], repeat=3)]


    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        Net1 = NeuralNetwork(self.weights_list, self.biases_list, self.functions)
        mystate_matrix = np.matrix(np.reshape(np.array(myState).flatten(), (27,1))) # get the board status as a matrix to be the input of NN
        shape = [27, 20, 27]
        moves = Net1.propagate(mystate_matrix)
        move = self.possible_moves[np.argmax(moves)]
        #print("\nmy move is: \n", move)
        return list(move)
        

    def getNN(self):
        return self.NNet


    @staticmethod
    def getSpec(self): # return the shape of the input neurons and the output neurons
        output_shape = np.array(27,27)

############################################################# Main (for testing) #####################################################################################        


#activation_functions = [Relu, Relu]
#shape = [3, 4, 3]
#w3,b3 = init_weights_biases(shape)
#Net1 = NeuralNetwork(w3, b3, activation_functions)
#output1 = Net1.propagate(X)
#print(output1)

