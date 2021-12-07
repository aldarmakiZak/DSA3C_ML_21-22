from numpy.lib.function_base import gradient
import CompromiseGame as cc
import Neural_Net as nn
import numpy as np
import random
import itertools

# generate a population of player objcets
def generate_players(players_no, shape, ac_func):
    my_players = []

    for i in range(players_no):
        w,b = nn.init_weights_biases(shape)
        my_players.append(nn.Zak_Player(w, b, ac_func))

    return my_players


def play_games(my_player, oppo_player, games_no): #function to play the game
    myWins = 0
    oppoWins = 0
    results_list = []
    for i in range(games_no):        
        game = cc.CompromiseGame(my_player, oppo_player, 30, 10)
        result = game.play()
        results_list.append(result)
        
        if result[0] > result[1]:
            myWins += 1
            #print("You won")
        elif result[0] < result[1]:
            oppoWins += 1
            #print("Computer won")
        elif result[0] == result[1]:
            print("draw")

        else:
            print("Something is wrong")
        #curses.wrapper(game.fancyPlay)
    #print("My wins: ", myWins)
    #print("Oppo wins: ", oppoWins)
    #winrate = myWins/games_no * 100
    #print(winrate)
    return results_list


# function to make each player in the population play x numbers of games
def population_play(players, games_no):
    population_result_list = []
    for i, player in enumerate(players):
        player_results = play_games(player, cc.RandomPlayer(), games_no)
        population_result_list.append(player_results)

    return population_result_list


# function to  calulate the total fitness of each player in the games
def calc_fitness(players, results):
    players_fitness = []

    for i, player in enumerate(results):
        player_score  = 0
        for game in player:
            score = game[0] - game[1]
            player_score += score
        players[i].fitness = player_score # store the fitness on the player object   
        players_fitness.append(player_score) 
    
    return players_fitness


# return sorted fitness and their previous indices
def sort_fitness(fitness_list):
    fitness_index = sorted(range(len(fitness_list)), key=lambda k:fitness_list[k])
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
def elitism(population, fitness, size_percentage= 5):
    sorted_fitnesses, sorted_indices = sort_fitness(fitness)
    elite_list = sorted_indices[-round(len(sorted_indices) * size_percentage/100) :]
    chosen_population = [population[x] for x in elite_list]
    return chosen_population


def crossover(parent1, parent2, mutation_rate=0.1):
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
    child_biasess = []
    for i in range(len(parent1_weights)):

        parent1_flattened_weights.append(parent1_weights[i].flatten())
        parent1_flattened_biases.append(parent1_biases[i].flatten())

        parent2_flattened_weights.append(parent2_weights[i].flatten())
        parent2_flattened_biases.append(parent2_biases[i].flatten())    

        weights_crossover_point = random.randint(0, len(parent1_flattened_weights[i])-1)
        biases_crossover_point = random.randint(0, len(parent1_flattened_biases[i])-1)
        
        #print(biases_crossover_point)
        child_flattened_weights.append(np.concatenate((parent1_flattened_weights[i][:weights_crossover_point], parent2_flattened_weights[i][weights_crossover_point:])))
        child_flattened_biases.append(np.concatenate((parent1_flattened_biases[i][:biases_crossover_point], parent2_flattened_biases[i][biases_crossover_point:])))

        child_biasess.append(np.array(child_flattened_biases[i]).reshape((parent1_biases[i].shape)))
        child_weights.append(np.array(child_flattened_weights[i]).reshape((parent1_weights[i].shape)))


    child = nn.Zak_Player(mutation(child_weights, mutation_rate), mutation(child_biasess, mutation_rate), [nn.Relu, nn.Relu])
    return child


def mutation(array_list, rate):    
    for i in array_list:
        mask = np.random.choice([0, 1], size=i.shape, p=((1 - rate), rate)).astype(bool) # crate a random masks array to be used in the mutation inspired from https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
        random_values = 0.10 * np.random.rand(i.shape[0],i.shape[1])
        i[mask] += random_values[mask]

    return array_list


def create_new_generation(population, fitness_list, gen_size):
    new_generation = []

    # get elite players from previous generation
    for i in elitism(population, fitness_list, 5):
        i.fitness = 0
        new_generation.append(i)

    while len(new_generation) < gen_size:
        parent1 = tournament_selection(population, 10)
        parent2 = tournament_selection(players, 10)
        child = crossover(parent1, parent2, 0.1)
        new_generation.append(child)
    
    return new_generation 


if __name__== "__main__":
    games_no = 10
    population_no = 15
    mutation_rate = 0.1

    activation_functions = [nn.Relu, nn.Relu]
    shape = [27, 3, 3]

    players = generate_players(population_no, shape, activation_functions)
    result_population = population_play(players, games_no)
    fitness = calc_fitness(players, result_population)
    
    new_gen = create_new_generation(players,fitness,15)
    new_gen_play = population_play(new_gen, games_no)
    new_gen_fitness = calc_fitness(new_gen, new_gen_play)
    #crossover(players[0], players[1], mutation_rate)
    #tournament_selection(players, 4)
    #sorted_fit, sorted_index = sort_fitness(fitness)
   # elitism(players, fitness, 5)
    #print(players[sorted_index[-1]].getNN().getLayers())
    #print(sorted_fit, "\n", sorted_index)
    #print(fitness[sorted_index[-1]])