import c4.gym_connect_four.envs.connect_four_env as gym
import c4.gym_connect_four.c4_utils as utils
import numpy as np
import random

#! Minimax
# Note: original_player = [original_player, original_maxPlayer]
def minimax(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth):
    if max_player:
        best_score = [-1, -999]

    else:
        best_score = [-1, 999]

    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 1)

    for action in env.ghost_available_moves(board):
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax(env, next_board, original_player, curr_player*-1, not max_player, depth+1)
        
        del next_board
        score[0] = action

        if max_player:
            if score[1] > best_score[1]:
                best_score = score

        else:
            if score[1] < best_score[1]:
                best_score = score

    return best_score

#! Minimax Prune
# Note: original_player = [original_player, original_maxPlayer]
def minimax_prune(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth, alpha, beta):
    if max_player:
        best_score = [-1, -999]

    else:
        best_score = [-1, 999]

    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 1)

    for action in env.ghost_available_moves(board):
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax_prune(env, next_board, original_player, curr_player*-1, not max_player, depth+1, alpha, beta)

        del next_board
        score[0] = action

        if max_player:
            best_score = utils.max_score(best_score, score)

            alpha = max(alpha, score[1])

            if beta < alpha:
                break

        else:
            best_score = utils.min_score(best_score, score)

            beta = min(beta, score[1])

            if beta < alpha:
                break

    return best_score

#! Minimax Heuristic
# Note: original_player = [original_player, original_maxPlayer]
def minimax_heuristic(env: gym.ConnectFourEnv, board, original_player, curr_player, max_player, depth):
    if max_player:
        best_score = [-1, -999]

    else:
        best_score = [-1, 999]

    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 5)

    if depth >= 6:
        best_score[1] = env.ghost_heuristic(board, original_player)
        return best_score

    for action in env.ghost_available_moves(board):
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax_heuristic(env, next_board, original_player, curr_player*-1, not max_player, depth+1)

        del next_board
        score[0] = action

        if max_player:
            if score[1] > best_score[1]:
                best_score = score

        else:
            if score[1] < best_score[1]:
                best_score = score

    return best_score

#! Minimax Prune Heuristic
# Note: original_player = [original_player, original_maxPlayer]
def minimax_prune_heuristic(env: gym.ConnectFourEnv, board, original_player, curr_player, maxPlayer, depth, alpha, beta):
    if maxPlayer:
        best_score = [-1, -999]

    else:
        best_score = [-1, 999]

    if env.ghost_is_terminal_state(board):
        return utils.score_calc(env, board, original_player, 5)

    if depth >= 6:
        best_score[1] = env.ghost_heuristic(board, original_player)
        return best_score

    for action in env.ghost_available_moves(board):
        next_board = env.ghost_step(board, action, curr_player)
        score = minimax_prune_heuristic(env, next_board, original_player, curr_player*-1, not maxPlayer, depth+1, alpha, beta)

        del next_board
        score[0] = action

        if maxPlayer:
            best_score = utils.max_score(best_score, score)

            alpha = max(alpha, score[1])

            if beta < alpha:
                break

        else:
            best_score = utils.min_score(best_score, score)

            beta = min(beta, score[1])

            if beta < alpha:
                break

    return best_score

#! Qlearn Act
def qlearnAct(env: gym.ConnectFourEnv, qtable, epsilon=0.4):
    available_moves = env.ghost_available_moves(env.board)

    if random.random() < epsilon:
        best_action = random.choice(list(available_moves))
    else:
        best_action = utils.find_best_action(
            qtable, env.board, available_moves)

    return best_action, epsilon - epsilon/50000


#! Qlearn Update
def qlearnUpdate(env: gym.ConnectFourEnv, qtable, prev_board, next_board, transition_action, score):
    lr = 0.5
    discount = 0.9
    qvalues = []

    tuple_prev_board = tuple(map(tuple, prev_board))

    if next_board is not None:
        available_actions = env.ghost_available_moves(next_board)
        tuple_next_board = tuple(map(tuple, next_board))

        for action in available_actions:
            qvalues.append(qtable[action][tuple_next_board])

        qtable[transition_action][tuple_prev_board] += lr*(score + discount*max(qvalues) - qtable[transition_action][tuple_prev_board])

    else:
        qtable[transition_action][tuple_prev_board] += lr*(score - qtable[transition_action][tuple_prev_board])

    return qtable
