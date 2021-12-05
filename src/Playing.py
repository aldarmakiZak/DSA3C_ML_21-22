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


def play_one_game(My_player, oppo_player, games_no):

    myWins = 0
    oppoWins = 0
    
    for i in range(games_no):        
        g = cc.CompromiseGame(My_player, oppo_player, 30, 10)
        result = g.play()
        if result[0] > result [1]:
            myWins += 1
            #print("You won")
        elif result[0] < result [1]:
            oppoWins += 1
            #print("Computer won")
        elif result[0] == result [1]:
            print("draw")

        else:
            print("Something is wrong")
        #curses.wrapper(g.fancyPlay)
    print ("My wins: ", myWins)
    print("Oppo wins: ", oppoWins)
    winrate = myWins/games_no * 100
    print(winrate)
    return winrate


# function to make each player in the population play x numbers of games
def play_games(players,games_no):
    wins = []
    for i,player in enumerate(players):
        print("\nplayer: ", i+1)
        game = play_one_game(player, cc.SmartGreedyPlayer(), games_no)
        wins.append(game)

    print(f"win rate for each player: {wins}")


if __name__== "__main__":
    games_no = 10
    population_no = 10
    
    activation_functions = [nn.Relu, nn.Relu]
    shape = [27, 50, 27]
    
    pB = cc.GreedyPlayer() 
    pC = cc.RandomPlayer()
    
    player = generate_players(population_no, shape, activation_functions)

    play_games(player, games_no)