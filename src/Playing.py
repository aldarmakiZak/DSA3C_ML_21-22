import CompromiseGame as cc
import Neural_Net as nn

games_no = 10

if __name__== "__main__":
    activation_functions = [nn.Relu, nn.Relu]
    shape = [27, 50, 27]
    w_l,b_l = nn.init_weights_biases(shape)
    pA = nn.Zak_Player(w_l, b_l, activation_functions)   #pA = nn.Zak_Player() 
    pB = cc.GreedyPlayer() 
    pC = cc.RandomPlayer()

    myWinRate = 0
    oppoWinRate = 0
    for i in range(games_no):
            
        g = cc.CompromiseGame(pA, pB, 30, 5)
        reslut = g.play()
        if reslut[0] > reslut [1]:
            myWinRate += 1
            print("You won")
        elif reslut[0] < reslut [1]:
            oppoWinRate += 1
            print("Computer won")
        elif reslut[0] == reslut [1]:
            print("draw")

        else:
            print("Something is wrong")
        #curses.wrapper(g.fancyPlay)
    print ("My win rate: ", myWinRate)
    print("Oppo win rate: ", oppoWinRate)