import gym_tictactoe.env as ttt_env


# alpha beta pruning heuristic idea:
# calculate the number of "key" positions held by the player (four corners and center)
# maybe also add negatives of all "key"  positions opponent has   
    
# If the maximizer has won, return +1
# Else if the minimzer has won, return -1
# Else if the state is 0, return 0
def score_calc(state, player, terminate_state):
    # if the current user of minimax (state[1]) has won, return score based on player

    # print(f"Terminate State: {terminate_state}")
    if (terminate_state > 0):
        if (ttt_env.tomark(terminate_state) == state[1]):
            if player == "max":
                return 1
            else:
                return -1
            
        if player == "max":
            return -1
        else:
            return 1
        
    return 0
        
def get_valid_actions(board):
    return [i for i, c in enumerate(board) if c == 0]

def take_action(state, action, curr_agent):
    board = list(state[0])
    board[action] = ttt_env.tocode(curr_agent)

    return (tuple(board), state[1])

def undo_action(state, action):
    board = list(state[0])
    board[action] = 0

    return (tuple(board), state[1])


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

def minimax(state, player, curr_agent):
    # if the current player is the maximizier
        # set the best score_action as [-1, -999]
    # else if the current player i the minimizer
        # set the best score_action as [-1, -999]

    if (player == "max"):
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # if the game is over (game state >= 0)
        # evaluate the current score (did O win, did X win or draw)
        # set the score_action as [-1, calc_score]

        #Note: which ever player wins, if its max, score is +1, if its min, score is -1


    if (ttt_env.check_game_status(state[0]) >= 0):
        score = [-1, score_calc(state, player, ttt_env.check_game_status(state[0]))]
        return score

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

    for action in get_valid_actions(state[0]):
        state = take_action(state, action, curr_agent)
        score = minimax(state, player, ttt_env.next_mark(curr_agent))
        state = undo_action(state, action)
        score[0] = action

        if player == "max":
            if score[1] > best_score[1]:
                best_score = score
        else:
            if score[1] < best_score[1]:
                best_score = score

    # return best_score
    return best_score