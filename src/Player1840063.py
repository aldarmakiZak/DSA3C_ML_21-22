# a file contains the best player trained to play the compromise game.. it takes the weights and biases from a file called best_player_attributes
# to use this player this file should be place within the folder of the neural network and compromise game python files
import numpy as np
import CompromiseGame as cc
import Neural_Net as nn
import itertools
import pickle

class NNPlayer():

    def load_best_attributes(self):
        
        player_attributes = pickle.load(self.file)
        self.weights_list = player_attributes["Weights"]
        self.biases_list = player_attributes["Biases"]
        self.functions = player_attributes["Activation_functions"]    
    
    def __init__(self):
        self.file = open("best_player_attributes", "rb")
        self.load_best_attributes()
        self.fitness = None
        self.games_won = None
        self.NNet = nn.NeuralNetwork(self.weights_list, self.biases_list, self.functions) 
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
    
 
