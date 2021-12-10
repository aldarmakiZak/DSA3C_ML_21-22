# This program is for training the compromise game player using genetic algorithms

import CompromiseGame as cc
import Neural_Net as nn
import numpy as np
import random
import copy
import pickle


# generate a population of player objcets
def generate_players(players_no, shape, ac_func):
    my_players = []

    for i in range(players_no):
        w,b = nn.init_weights_biases(shape)
        my_players.append(nn.NNPlayer(w, b, ac_func))

    return my_players


# function to play number of games for each player
def play_games(my_player, oppo_player, games_no):
    myWins = 0
    oppoWins = 0
    results_list = []
    for i in range(games_no):        
        game = cc.CompromiseGame(my_player, oppo_player, 30, 10)
        result = game.play()
        results_list.append(result)
        
        if result[0] > result[1]:
            myWins += 1

        elif result[0] < result[1]:
            oppoWins += 1

        elif result[0] == result[1]:
            print("draw")

        else:
            print("Something is wrong")
    my_player.games_won = myWins

    return results_list


# function to make each player in a population play x numbers of games
def population_play(players, games_no):
    population_result_list = []
    for i, player in enumerate(players):
        player_results = play_games(player, cc.RandomPlayer(), games_no)
        population_result_list.append(player_results)

    return population_result_list


# function to calulate the total fitness of each player in the games
def calc_fitness(players, results):
    players_fitness = []

    for i, player in enumerate(results):
        player_score  = 0
        for game in player:
            score = game[0] - game[1]
            player_score += score
        players[i].fitness = player_score # store the fitness on the player object   
        players[i].fitness += players[i].games_won * 30
        players_fitness.append(player_score + players[i].games_won * 30) 
    
    return players_fitness


# return sorted fitness and their indices indices
def sort_fitness(fitness_list):
    fitness_sorted = np.sort(fitness_list)
    fitness_sorted_indices = np.argsort(fitness_list)
    return fitness_sorted, fitness_sorted_indices


# get the best parent based on tournament selection 
def tournament_selection(population_list, size):
    parent = None

    for i in range(size):
        random_choice = random.choice(population_list)

        if parent is None or int(random_choice.fitness) >= parent.fitness :
            parent = random_choice

    return parent


#Get the fittest individuals from the population to be passed to the next generation 
def elitism(population, fitness, size_percentage= 20):
    sorted_fitnesses, sorted_indices = sort_fitness(fitness)
    elite_list = sorted_indices[-round(len(sorted_indices) * size_percentage/100) :]
    chosen_population = [population[x] for x in elite_list]
    return chosen_population


# this function gets the two players weights and biases and return a new player from the combination of both 
def crossover(parent1, parent2, activation_functions, mutation_rate=0.1):
    parent1_weights = parent1.getNN().get_weights()
    parent1_biases = parent1.getNN().get_biases()
    
    parent2_weights = parent2.getNN().get_weights()
    parent2_biases = parent2.getNN().get_biases()
    
    parent1_flattened_weights = []
    parent1_flattened_biases = []
    
    parent2_flattened_weights = []
    parent2_flattened_biases = []

    child_flattened_weights = []
    child_flattened_biases = []
    
    child_weights = []
    child_biases = []
    for i in range(len(parent1_weights)):

        parent1_flattened_weights.append(parent1_weights[i].flatten()) # flatten the weights to be 1d array
        parent1_flattened_biases.append(parent1_biases[i].flatten())

        parent2_flattened_weights.append(parent2_weights[i].flatten())
        parent2_flattened_biases.append(parent2_biases[i].flatten())    

        weights_crossover_point = random.randint(0, len(parent1_flattened_weights[i])-1) # choose a random number to be the crossover point for weights
        biases_crossover_point = random.randint(0, len(parent1_flattened_biases[i])-1)  # choose a random number to be the crossover point for biases
        
        child_flattened_weights.append(np.concatenate((parent1_flattened_weights[i][:weights_crossover_point], parent2_flattened_weights[i][weights_crossover_point:]))) #get the first portion of the weights from paren1 and the second from parent2
        child_flattened_biases.append(np.concatenate((parent1_flattened_biases[i][:biases_crossover_point], parent2_flattened_biases[i][biases_crossover_point:]))) #get the first portion of the biases from paren1 and the second from parent2

        child_biases.append(np.array(child_flattened_biases[i]).reshape((parent1_biases[i].shape)))
        child_weights.append(np.array(child_flattened_weights[i]).reshape((parent1_weights[i].shape)))


    child = nn.NNPlayer(mutation(child_weights, mutation_rate), mutation(child_biases, mutation_rate), activation_functions)
    return child


# asexual reproduction.. copy the parent weights and biases to the child
def create_child_copy(parent, activation_functions, mutation_rate=0.1): 
    child_weights = parent.getNN().get_weights()
    child_biases = parent.getNN().get_biases()

    child = nn.NNPlayer(mutation(child_weights, mutation_rate), mutation(child_biases, mutation_rate), activation_functions)
    return child


def mutation(arrays_list, rate):    
    for i in arrays_list:
        mask = np.random.choice([0, 1], size=i.shape, p=((1 - rate), rate)).astype(bool) # crate a random masks array to be used in the mutation inspired from https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
        random_values = np.random.rand(i.shape[0],i.shape[1])
        i[mask] += random_values[mask] # add from the random values based on the mask

    return arrays_list


# function to create a new generation for the genetic algorithms 
def create_new_generation(population, fitness_list, gen_size, activation_functions, crossover_rate, mutation_rate):
    new_generation = []
    
    # get elite players from previous generation
    for i in elitism(population, fitness_list, 20):
        i.fitness = 0
        i.games_won = 0
        new_generation.append(i)

    # create the children
    while len(new_generation) < gen_size:
        random_index = random.uniform(0, 1)
        parent1 = tournament_selection(population, 4) # select the parents based on tournament selection
        parent2 = tournament_selection(population, 4)
        
        if random_index <= crossover_rate: # choose which method to use for breading
            child = crossover(parent1, parent2, activation_functions,mutation_rate)
        else:
            child = create_child_copy(parent1, activation_functions, mutation_rate)
        
        new_generation.append(child)
    
    return new_generation 


# function to run the evolution cycles
def create_evolution(population_size, generations_number, games_num, crossover_rate, mutation_rate):
    activation_functions = [nn.Relu, nn.Relu]
    shape = [27,100, 27]

    initial_population = generate_players(population_size, shape, activation_functions) #create the initial population using ramdom weights
    initial_population_results = population_play(initial_population, games_num)
    fitness = calc_fitness(initial_population, initial_population_results)
    generation_no = 1
    print_info(initial_population, fitness, generation_no, games_num)

    best_player = None

    while generation_no < generations_number:
        generation_no += 1
        population = initial_population
        sorted_fitness, sorted_incides = sort_fitness(fitness)
        
        if best_player is None or best_player.fitness < population[sorted_incides[-1]].fitness: # check if we have a new best player

            best_player = copy.deepcopy(population[sorted_incides[-1]])
            print(f"Best Player Fitness: {best_player.fitness}")
            save_best_player(population[sorted_incides[-1]])

        new_population = create_new_generation(population,fitness,population_size, activation_functions, crossover_rate, mutation_rate)
        game_result = population_play(new_population, games_num)
        new_fitness = calc_fitness(new_population, game_result)
        print_info(new_population, new_fitness, generation_no, games_num)
        initial_population = new_population
        fitness = new_fitness


# function to print the information of the training 
def print_info(population, fitness, generation, games_number):
    sorted_fitness, sorted_incides = sort_fitness(fitness)
    wins = []
    for i in population: # calculate the win rate for the generation 
        wins.append(i.games_won/games_number *100)

    wins_avg = sum(wins)/len(wins)
    print("Generation number: \t", generation)
    print("Best player fitness: \t", population[sorted_incides[-1]].fitness)
    print("Best player won games: \t", population[sorted_incides[-1]].games_won)
    print("Best player Avg win rate: \t", (population[sorted_incides[-1]].games_won)/games_number *100)
    print("Population win rate", wins_avg)
    print("\n\n")


# save the best player weights and biasess to a file
def save_best_player(player): 
    player_weights = player.getNN().get_weights()
    player_biases = player.getNN().get_biases()
    player_activation = player.getNN().functions
    palyer_attributes = {"Weights": player_weights,
                        "Biases": player_biases,
                        "Activation_functions": player_activation}

    player_file = open("testing_players/best_randn_random_player", "wb")
    pickle.dump(palyer_attributes, player_file)
    player_file.close()


# load the attributes of best player
def load_best_player(player_file):
    f = open(player_file,"rb")
    player_attributes = pickle.load(f)
    best_player = nn.NNPlayer(player_attributes["Weights"], player_attributes["Biases"], player_attributes["Activation_functions"])
    best_player
    f.close()

    return(best_player)


# function to save the last generation of training model to a file -for testing-
def save_last_generation(population):
    generation_list = []
    for player in population:
        player_weights = player.getNN().get_weights()
        player_biases = player.getNN().get_biases()
        player_activation = player.getNN().functions
        player_attributes = {"Weights": player_weights,
                        "Biases": player_biases,
                        "Activation_functions": player_activation}
        generation_list.append(player_attributes)

    generation_file = open("best_random_generation", "wb")
    pickle.dump(generation_list, generation_file)
    generation_file.close()



if __name__== "__main__":
    games_no = 100
    population_no = 50
    generations_no = 5000 
    mutation_rate = 0.03
    crossover_rate = 0.9
    
    create_evolution(population_no, generations_no, games_no, crossover_rate, mutation_rate)
