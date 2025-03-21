import gym_tictactoe.env as ttt_env


# alpha beta pruning heuristic idea:
# calculate the number of "key" positions held by the player (four corners and center)
# maybe also add negatives of all "key"  positions opponent has


# #! Next Plater
# def next_player(curr_player):
#     return "min" if curr_player == "max" else "max"


# #! Tictactoe Heuristic
# def tictactoe_heursitic(state, player, curr_agent):
#     num_key_pos = 0

#     key_pos = [1, 3, 5, 7, 9]

#     for pos in range(10):
#         if (state[0][pos] == ttt_env.tocode(curr_agent)) and (pos % 2 != 0):
#             if player == "max":
#                 key_positions += 1
#             else:
#                 key_positions -= 1


# Alpha-Beta Pruning

def max_score(best_score, score):
    if (best_score[1] > score[1]):
        return best_score
    else:
        return score
    

def min_score(best_score, score):
    if (best_score[1] < score[1]):
        return best_score
    else:
        return score


#! Score Calc
# Calculate the Minimax score of a path, given the original agent, their player and the termination status of the game
def score_calc(state, terminate_state):
    # If the game has terminated with an agent victory
    if (terminate_state > 0):
        if (ttt_env.tomark(terminate_state) == 'O'):
            if (state[1] == 'O'):
                if (state[2]):
                    return 1
                else:
                    return -1
            else:
                if (state[2]):
                    return -1
                else:
                    return 1
        else:
            if (state[1] =='O'):
                if (state[2]):
                    return -1
                else:
                    return 1
            else:
                if (state[2]):
                    return 1
                else:
                    return -1
        
        #if the O has won, return 1
        #else the X has won, return -1

    # else return 0


        # # If the original agent has won
        # if (ttt_env.tomark(terminate_state) == state[1]):
        #     # If the agent is playing as maximizer, return score = 1, else return score = -1
        #     if maxPlayer:
        #         return 1
        #     else:
        #         return -1

        # # Else, the opponent must have won
        # # If the agent is playing as maximizer, return score = -1, else return score = 1
        # if maxPlayer:
        #     return -1
        # else:
        #     return 1

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
    return (tuple(board), state[1], state[2])


#! Undo Action
# Undo an action taken on the board, given the state of the board and the action
def undo_action(state, action):
    # Store the board state as a list
    board = list(state[0])
    # Update the board to undo the action
    board[action] = 0

    # Return the correct format of the state (board as tuple, original agent)
    return (tuple(board), state[1], state[2])