import random
import ttt.gym_tictactoe.env as ttt_env


#! Max Score
# Get the max score of two arrays
def max_score(score1, score2):
    return score1 if score1[1] > score2[1] else score2


#! Min Score
# Get the min score of two arrays
def min_score(score1, score2):
    return score1 if score1[1] < score2[1] else score2


#! Score Calc
# Calculate the best score given the current player
def score_calc(max_player, terminate_state):
    # The player opposite to curr_player has won or its a draw
    # Check if a draw, if not, return a score using the loser's max_player status (opposite score because winner assumed to be opposite max_player)
    if (terminate_state > 0):
        if (max_player):
            return [-1, -1]
        else:
            return [-1, 1]
    else:
        return [-1, 0]
        

#! Get Valid Actions
# Get all available actions on a board
def get_valid_actions(board):
    # Return all available actions
    return [i for i, c in enumerate(board) if c == 0]


#! Take Action
# Update a board to reflect a taken action by the current agent
def take_action(state, action, curr_agent):
    board = list(state[0])
    board[action] = ttt_env.tocode(curr_agent)

    # Return the state object wit hthe updated board
    return (tuple(board), state[1], state[2])


#! Find Best Action
# Get the best action using a Qtable
def find_best_action(state, ava_actions, qtable):
    # Get all Qvalues for a given state and available actions
    qvalues = [qtable[action][state[0]] for action in ava_actions]

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
        best_action = random.choice(ava_actions)
    else:
        best_idx = random.choice(best_indexes)
        best_action = ava_actions[best_idx]

    # Return the best action
    return best_action
