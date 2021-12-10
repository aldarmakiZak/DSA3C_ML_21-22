
# This program is the structure of the neural network and the player class

import numpy as np
from types import FunctionType
import itertools
import copy


#activation functions
def Relu(inputs):
    output = np.maximum(0, inputs)
    return output


def Soft_max(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=0))
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


# function to create a random lists of shapes and biasses based on the shape of the network
def init_weights_biases(net_shape, activation_func=None):
    weights = []
    biases = []
    for i,j in zip(net_shape,net_shape[1:]):
        weight =  np.random.randn(j, i)
        bias =  np.random.randn(j, 1)
        weights.append(weight)
        biases.append(bias)
    
    return weights, biases


# class to calculate a layer of a neural network
class Layer:
    
    def __init__(self, weights: np.matrix, biases: np.matrix, activation_func: FunctionType ):
        if weights.shape[0] != len(biases):
            raise ValueError(" ValueError the shape of the layer inputs in wrong")

        self.weights = copy.deepcopy(weights)
        self.biases = copy.deepcopy(biases)
        self.activation_func = activation_func
        

    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases
        return self.activation_func(self.output)

    def getMatrix(self):
        return self.weights

    def getBiasVector(self):
        return self.biases

    def getFunction(self):
        return self.activation_func



class NeuralNetwork:
    def __init__(self, weights_list: np.array, biases_list: np.array, functions):
        if len(weights_list) != len(biases_list) or len(biases_list) != len(functions):
            raise ValueError(" ValueError the shapes of the neural network inputs is not valid ")
        
        self.weights_list = weights_list
        self.biases_list = biases_list
        self.functions = functions
        self.layers = []

        for weights, biases, activation_function in list(zip(self.weights_list, self.biases_list, self.functions)): # Create layer from the weights and biases lists
             self.layers.append(Layer(weights, biases, activation_function))

    def propagate(self, inputs):
        prev_outputs = inputs
        for layer in self.layers:
            layer_output = layer.forward(prev_outputs)
            if layer_output.shape[0] != layer.weights.shape[0]:
                raise ValueError(" ValueError the shapes of the neural network inputs is not valid ")
            
            prev_outputs = layer_output
        return prev_outputs  

    def getLayers(self):
        return self.layers

    # return weights to be flattened in the genetic algorithms
    def get_weights(self):
        return self.weights_list 
    
    # return biases to be flattened in the genetic algorithms
    def get_biases(self):
        return self.biases_list 


### class player to play the game
class NNPlayer():
       
    def __init__(self, weights_list, biases_list, functions):
        self.weights_list = weights_list
        self.biases_list = biases_list
        self.functions = functions #[Relu, Relu]
        self.fitness = None
        self.games_won = None
        self.NNet = NeuralNetwork(self.weights_list, self.biases_list, self.functions) 
        self.possible_moves = [p for p in itertools.product([0, 1, 2], repeat=3)] # get all the combination of the possible moves

    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        my_state_matrix = np.matrix(np.reshape(np.array(myState).flatten(), (27,1))) # get the board status as a matrix to be the input of NN
        oppo_state_matrix = np.matrix(np.reshape(np.array(oppState).flatten(), (27,1))) # get the board status as a matrix to be the input of NN

        moves = self.NNet.propagate(my_state_matrix-oppo_state_matrix)
        move = self.possible_moves[np.argmax(moves)]
        return list(move)
        

    def getNN(self):
        return self.NNet


    @staticmethod
    def getSpecs(): # return the shape of the input neurons and the output neurons
        return (27,27)