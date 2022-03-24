# Compromise Game Machine Learning Player

This code is for a player that uses Neural Networks and Genitic Algorithms to play a simple board game created specifically for machine learning. the game and its rules can be found in https://github.com/gmoutsin/Compromise. 




## Code specifications



### Layer

A class called `Layer` That has the functionality of a Neural Network layer  

Its constructor accepts a numpy matrix, which represents the weights of the layer, a numpy array, which represents the biases of the layer, and a function, which represents the activation function. The constructor checks whether the dimensions of weights and biases are compatible and raise a `ValueError` exception if not.

The constructor copys the matrix and the array, so that they cannot be changed externally.

The class contains a method called `forward`, which takes the input to the layer in the form of a numpy array, and returns the output of the layer which is a numpy array.

The class also contains methods called `getMatrix`, `getBiasVector` and `getFunction` which accept no argument and returns the matrix, bias and function respectively.

### Neural network

This class has the functionality of a neural network.

Its constructor accepts a list of matrices, a list of arrays and a list of functions. Then it takes consecutively one from each and create the layers of the neural network. The constructor also checks that all three lists have the same length and that the output of a layer can be used as an input to the next layer and raise a `ValueError` exception if this is not the case.

The class contains a method called `propagate`, which takes the input to the neural network, as a numpy array, and returns the output, as a numpy array.

The class also contains a method called `getLayers`, which accepts no arguments and returns a list containing the layers of the neural network.

### Player

A class called `NNPlayer` whith contains the functionality of the game player.

Its constructor accepts the same arguments as the NeuralNetwork constructor, i.e. a list of matrices, a list of arrays and a list of functions.

The class also contains a method called `play`, which takes the state of the game (see CompromiseGame for more information) and returns a valid move, i.e. a list of length 3 whose elements are from the set `{0,1,2}`.

The class contains a method called `getNN`, which accepts no arguments and returns the neural network of the player.

The class also contains a static method called `getSpecs`, that accepts no arguments and returns a 2-tuple. The first number in the tuple is the dimension of the input to the neural network and the second number of the tuple is the dimension of the output of the neural network. This method is essential for the tests to work.

### Fitness Function

The fitness for the neural network outputs for a player is based on two factors which are:

1 – The first factor to determine the fitness is the score that a player gets from each game.
This is calculated by getting the sum of the deferences between the neural network player
score and the opponent for a set of games.

2 – The number of games a player wins against the opponent: for each game the neural
network player wins, it will be rewarded with additional 20 fitness points.








