import CompromiseGame as cg
import Neural_Net as nn
import Genetic_Algorithm as ga
import pickle


#f = open("best_smart_greedy_player", "rb")
f = open("testing_players/best_50popsize_random_player", "rb")
player1_attributes = pickle.load(f)

# f2 = open("testing_players/best_greedy_player", "rb")
# player2_attributes = pickle.load(f2)
best_player1 = nn.NNPlayer(player1_attributes["Weights"], player1_attributes["Biases"], player1_attributes["Activation_functions"])
#best_player2 = nn.NNPlayer(player2_attributes["Weights"], player2_attributes["Biases"], player2_attributes["Activation_functions"])

f.close()
#f2.close()
#results = ga.population_play([best_player1, best_player2], 10)
#results = ga.play_games(best_player1, cg.RandomPlayer(), 100)
results = ga.play_games(best_player1, cg.RandomPlayer(), 100)


games_won = 0
for i in range(len(results)):
    
    if  results[i][0] > results[i][1]:
        games_won += 1

    

#fitness = ga.calc_fitness([best_player1, best_player2], pop)
print(games_won)
#print(fitness)

