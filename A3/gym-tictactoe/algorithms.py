import gym_tictactoe.env as ttt_env
     
    

    # If the maximizer has won, return +1
    # Else if the minimzer has won, return -1
    # Else if the state is 0, return 0

#! Minimax
# Two opponents; a maximizer player and a minimizer player

# Maximizer has all victories be considered a +1
# Minimizer has all victories be considered a -1
# Draws are considered 0 (no victory for anyone)

# For a player:
# Given the state of the board, provided is has not terminated:
# Check the available moves and build a tree of possible moves until termination in each case (victory or draw)
# If the returned score is better than the current score, that path is the best path for the agent

# state = ((board_state), mark)

def minimax(state, player):
    print("Empty")
    # if the current player is the maximizier
        # set the best score_action as [-1, -999]
    # else if the current player i the minimizer
        # set the best score_action as [-1, -999]

    # if the game is over (game state >= 0)
        # evaluate the current score (did O win, did X win or draw)
        # set the score_action as [-1, calc_score]

        #Note: which ever player wins, if its max, score is +1, if its min, score is -1

        # for every available move left:
            # take the given move for the current agent
            # call minimax, giving the opposite agent to simulate back and forth moves
            # this will return the score
            # undo the action
            
            # if the agent is max:
                # if score > best_score:
                    # best_score = score
            # elif the agent is min:
                # if score < best_score:
                    # best_score = score

    # return best_score