from numpy.lib.function_base import gradient
import CompromiseGame as cc
import Neural_Net as nn


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
        #print("\nplayer: ", i+1)
        player_results = play_games(player, cc.RandomPlayer(), games_no)
        population_result_list.append(player_results)

    return population_result_list


# function to  calulate the total fitness of each player in the games and sort it
def calc_fitness(results):
    player_fitness = []


    for player in results:
        player_score  = 0
        for game in player:
            score = game[0] - game[1]
            player_score += score            
        player_fitness.append(player_score)
    
    print("Players fitness: ", player_fitness)
    print("Sorted fitness: ", sorted(player_fitness))
    
    print("index of players: ", sorted(range(len(player_fitness)), key=lambda k:player_fitness[k]))
    return player_fitness

if __name__== "__main__":
    games_no = 10
    population_no = 10

    activation_functions = [nn.Relu, nn.Relu]
    shape = [27, 50, 27]

    players = generate_players(population_no, shape, activation_functions)
    result_population = population_play(players, games_no)
    calc_fitness(result_population)