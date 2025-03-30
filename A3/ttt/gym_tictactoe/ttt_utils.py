import random
import ttt.gym_tictactoe.env as ttt_env


#! Max Score
def max_score(score1, score2):
    return score1 if score1[1] > score2[1] else score2
    

def min_score(score1, score2):
    return score1 if score1[1] < score2[1] else score2

#! Score Calc
# Calculate the Minimax score of a path, given the original agent, their player and the termination status of the game
def score_calc(state, terminate_state):
    # If the game has terminated with an agent victory
    if (terminate_state > 0):
        if (ttt_env.tomark(terminate_state) == state[1]):
            if (state[2]):
                return [-1, 1]
            else:
                return [-1, -1]
        else:
            if (state[2]):
                return [-1, -1]
            else:
                return [-1, 1]
    else:
        return [-1, 0]
        

#! Get Valid Actions
# Get all available actions for the agent, given the state of the board
def get_valid_actions(board):
    # Return all available actions
    return [i for i, c in enumerate(board) if c == 0]


#! Take Action
# Update the board to reflect an agent's action, given the state of the board, the action and the current agent
def take_action(state, action, curr_agent):
    # Update the board with the action, indicating the current agent is taking it
    board = list(state[0])
    board[action] = ttt_env.tocode(curr_agent)

    # Return the correct format of the state (board as tuple, original agent)
    return (tuple(board), state[1], state[2])


#! Undo Action
# Undo an action taken on the board, given the state of the board and the action
def undo_action(state, action):
    # Update the board to undo the action
    board = list(state[0])
    board[action] = 0

    # Return the correct format of the state (board as tuple, original agent)
    return (tuple(board), state[1], state[2])


def find_best_action(state, ava_actions, qtable):
    qvalues = [qtable[action][state[0]] for action in ava_actions]

    best_value = 0
    best_indexes = []

    for idx in range(0, len(qvalues)):
        if qvalues[idx] > best_value:
            best_value = qvalues[idx]
            best_indexes = [idx]
        
        elif qvalues[idx] == best_value:
            best_indexes.append(idx)

    if (len(best_indexes) == 0):
        best_action = random.choice(ava_actions)
    else:
        best_idx = random.choice(best_indexes)
        best_action = ava_actions[best_idx]

    return best_action