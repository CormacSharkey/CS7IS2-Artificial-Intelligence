import c4.gym_connect_four.envs.connect_four_env as gym
import c4.gym_connect_four.c4_utils as utils
import numpy as np
import random


# * Note: original_player = [original_player, original_maxPlayer]


#! Minimax Algorithm
def minimax(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth):
    # Best score depends on max_player
    if max_player:
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # If the game is over, return the state score
    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 1)

    # Take every available action as the current agent, then recursively use minimax again
    for action in env.ghost_available_moves(board):

        # Take an action and call minimax
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax(env, next_board, original_player, curr_player*-1, not max_player, depth+1)
        del next_board

        # Update the returned score
        score[0] = action

        # Updating the best score depends on max_player
        if max_player:
            if score[1] > best_score[1]:
                best_score = score

        else:
            if score[1] < best_score[1]:
                best_score = score

    # Return the best score
    return best_score


#! Minimax Alpha Beta Prune Algorithm
def minimax_prune(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth, alpha, beta):
    # Best score depends on max_player
    if max_player:
        best_score = [-1, -999]

    else:
        best_score = [-1, 999]

    # If the game is over, return the state score
    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 1)

    # Take every available action as the current agent, then recursively use minimax again
    for action in env.ghost_available_moves(board):

        # Take an action and call minimax prune
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax_prune(env, next_board, original_player, curr_player*-1, not max_player, depth+1, alpha, beta)
        del next_board

        # Update the returned score
        score[0] = action

        # Updating the best score depends on max_player
        if max_player:
            # Get the best score and compare it to alpha
            best_score = utils.max_score(best_score, score)
            alpha = max(alpha, score[1])
            if beta < alpha:
                break
        else:
            # Get the best score and compare it to beta
            best_score = utils.min_score(best_score, score)
            beta = min(beta, score[1])
            if beta < alpha:
                break

    # Return the best score
    return best_score


#! Minimax Heuristic Algorithm
def minimax_heuristic(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth):
    # Best score depends on max_player
    if max_player:
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # If the game is over, return the state score
    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 5)

    # If the depth-limit has been reached, heuristically calculate a score
    if depth >= 5:
        best_score[1] = env.ghost_heuristic(board, original_player)
        return best_score

    # Take every available action as the current agent, then recursively use minimax again
    for action in env.ghost_available_moves(board):

        # Take an action and call minimax
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax_heuristic(env, next_board, original_player, curr_player*-1, not max_player, depth+1)
        del next_board

        # Update the returned score
        score[0] = action

        # Updating the best score depends on max_player
        if max_player:
            if score[1] > best_score[1]:
                best_score = score
        else:
            if score[1] < best_score[1]:
                best_score = score

    # Return the best score
    return best_score

#! Minimax Alpha Beta Prune Heuristic Algorithm
def minimax_prune_heuristic(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth, alpha, beta):
    # Best score depends on max_player
    if max_player:
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # If the game is over, return the state score
    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 5)

    # If the depth-limit has been reached, heuristically calculate a score
    if depth >= 5:
        best_score[1] = env.ghost_heuristic(board, original_player)
        return best_score

    # Take every available action as the current agent, then recursively use minimax again
    for action in env.ghost_available_moves(board):
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax_prune_heuristic(env, next_board, original_player, curr_player*-1, not max_player, depth+1, alpha, beta)
        del next_board

        # Update the returned score
        score[0] = action

        # Updating the best score depends on max_player
        if max_player:
            # Get the best score and compare it to alpha
            best_score = utils.max_score(best_score, score)
            alpha = max(alpha, score[1])
            if beta < alpha:
                break
        else:
            # Get the best score and compare it to beta
            best_score = utils.min_score(best_score, score)
            beta = min(beta, score[1])
            if beta < alpha:
                break

    # Return the best score
    return best_score

#! Qlearn Act Algorithm
def qlearnAct(env: gym.ConnectFourEnv, qtable, epsilon=0.4):
    # Get the available actions
    available_moves = env.ghost_available_moves(env.board)

    # Exploration vs. Exploitation - random exploration vs. Qtable decision
    if random.random() < epsilon:
        best_action = random.choice(list(available_moves))
    else:
        best_action = utils.find_best_action(
            qtable, env.board, available_moves)

    # Return the best action and decayed epsilon
    return best_action, epsilon - epsilon/50000


#! Qlearn Update Algorithm
def qlearnUpdate(env: gym.ConnectFourEnv, qtable, prev_board, next_board, transition_action, score):
    # Declare parameters
    lr = 0.5
    discount = 0.9
    qvalues = []

    tuple_prev_board = tuple(map(tuple, prev_board))

    # If this isn't the terminal update, update with next_state
    if next_board is not None:
        available_actions = env.ghost_available_moves(next_board)
        tuple_next_board = tuple(map(tuple, next_board))

        for action in available_actions:
            qvalues.append(qtable[action][tuple_next_board])

        qtable[transition_action][tuple_prev_board] += lr*(score + discount*max(qvalues) - qtable[transition_action][tuple_prev_board])

    else:
        qtable[transition_action][tuple_prev_board] += lr*(score - qtable[transition_action][tuple_prev_board])

    # Return the Qtable
    return qtable
