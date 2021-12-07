from numpy.lib.function_base import gradient
import CompromiseGame as cc
import Neural_Net as nn
import numpy as np
import random

# generate a population of player objcets
def generate_players(players_no, shape, ac_func):
    my_players = []

    for i in range(players_no):
        w,b = nn.init_weights_biases(shape)
        my_players.append(nn.Zak_Player(w, b, ac_func))

    return my_players


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


# function to  calulate the total fitness of each player in the games and sort it
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
def elitism(population, fitness, size= 5):
    sorted_fitnesses, sorted_indices = sort_fitness(fitness)
    elite_list = sorted_indices[-size:]
    chosen_population = [population[x] for x in elite_list]
    return chosen_population


def crossover(parent1, parent2, rate):
    pass

def mutation(individual, rate):
    pass

if __name__== "__main__":
    games_no = 10
    population_no = 15

    activation_functions = [nn.Relu, nn.Relu]
    shape = [27, 50, 27]

    players = generate_players(population_no, shape, activation_functions)
    result_population = population_play(players, games_no)
    fitness = calc_fitness(players, result_population)
    tournament_selection(players, 4)
    #sorted_fit, sorted_index = sort_fitness(fitness)
   # elitism(players, fitness, 5)
    #print(players[sorted_index[-1]].getNN().getLayers())
    #print(sorted_fit, "\n", sorted_index)
    #print(fitness[sorted_index[-1]])