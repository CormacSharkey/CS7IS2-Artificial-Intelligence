import gym_tictactoe.env as ttt_env


# alpha beta pruning heuristic idea:
# calculate the number of "key" positions held by the player (four corners and center)
# maybe also add negatives of all "key"  positions opponent has  

#* Note: state = ((board_state), mark)  


#! Score Calc
# Calculate the Minimax score of a path, given the original agent, their player and the termination status of the game
def score_calc(state, player, terminate_state):
    # If the game has terminated with an agent victory
    if (terminate_state > 0):
        # If the original agent has won
        if (ttt_env.tomark(terminate_state) == state[1]):
            # If the agent is playing as maximizer, return score = 1, else return score = -1
            if player == "max":
                return 1
            else:
                return -1
            
        # Else, the opponent must have won
        # If the agent is playing as maximizer, return score = -1, else return score = 1
        if player == "max":
            return -1
        else:
            return 1

    # Else the game must have terminated as a draw, return score = 0
    return 0


#! Get Valid Actions
# Get all available actions for the agent, given the state of the board
def get_valid_actions(board):
    # For every id and state value on the board, return (in a list) all ids where the state = 0 (empty state)
    return [i for i, c in enumerate(board) if c == 0]


#! Take Action
# Update the board to reflect an agent's action, given the state of the board, the action and the current agent
def take_action(state, action, curr_agent):
    # Store the board state as a list
    board = list(state[0])
    # Update the board with the action, indicating the current agent is taking it
    board[action] = ttt_env.tocode(curr_agent)

    # Return the correct format of the state (board as tuple, original agent)
    return (tuple(board), state[1])


#! Undo Action
# Undo an action taken on the board, given the state of the board and the action
def undo_action(state, action):
    # Store the board state as a list
    board = list(state[0])
    # Update the board to undo the action
    board[action] = 0

    # Return the correct format of the state (board as tuple, original agent)
    return (tuple(board), state[1])


#! Minimax
# Minimax Algorithm - pre-calculates all possible final states from the current board state, and determines the best action to make to ensure victory
# Works optimally when the opponent is playing optimally (max agent vs. min agent)
def minimax(state, player, curr_agent):

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