import c4.gym_connect_four.envs.connect_four_env as gym
import random


#! Max Score
# GEt the mac score of the two arrays
def max_score(score1, score2):
    return score1 if score1[1] > score2[1] else score2


#! Min Score
# Get the min score of two arrays
def min_score(score1, score2):
    return score1 if score1[1] < score2[1] else score2


#! Find Best Action
# Get the best action using a Qtable
def find_best_action(qtable, board, available_moves):
    # Get all Qvalues for a given state and available actions
    tuple_board = tuple(map(tuple, board))
    qvalues = [qtable[action][tuple_board] for action in list(available_moves)]

    best_value = 0
    best_indexes = []

    # For every Qvalue, determine the best Qvalue(s) and store them
    for idx in range(0, len(qvalues)):
        if qvalues[idx] > best_value:
            best_value = qvalues[idx]
            best_indexes = [idx]

        elif qvalues[idx] == best_value:
            best_indexes.append(idx)

    # If there are no best Qvalues (> 0), pick a random action
    if (len(best_indexes) == 0):
        best_action = random.choice(list(available_moves))
    else:
        best_idx = random.choice(best_indexes)
        best_action = list(available_moves)[best_idx]

    # Return the best action
    return best_action


#! Score Calc
# Calculate the best score given the current player
def score_calc(env: gym.ConnectFourEnv, board, max_player, weight):
    # # If the game has terminated with an agent victory
    # if env.ghost_check_winner(board) == curr_player*-1:
    #         # The best score depends on max_player
    #     if max_player:
    #         return [-1, weight]
    #     else:
    #         return [-1, -weight]
    # # If the game has terminated with an opponent victory
    # elif env.ghost_check_winner(board) == curr_player:

    # The player opposite to curr_player has won or its a draw
    # Check if a draw, if not, return a score using the loser's max_player status (opposite score because winner assumed to be opposite max_player)
    if env.ghost_check_winner(board) != 0:
        if max_player:
            return [-1, -weight]
        else:
            return [-1, weight]
    else:
        return [-1, weight/2]
