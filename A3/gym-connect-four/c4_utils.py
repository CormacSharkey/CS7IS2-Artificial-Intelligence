import gym_connect_four.envs.connect_four_env as gym
import random


def max_score(score1, score2):
    return score1 if score1[1] > score2[1] else score2

def min_score(score1, score2):
    return score1 if score1[1] < score2[1] else score2

def find_best_action(qtable, board, available_moves):
    tuple_board = tuple(map(tuple, board))
    qvalues = [qtable[action][tuple_board] for action in list(available_moves)]

    best_value = 0
    best_indexes = []

    for idx in range(0, len(qvalues)):
        if qvalues[idx] > best_value:
            best_value = qvalues[idx]
            best_indexes = [idx]

        elif qvalues[idx] == best_value:
            best_indexes.append(idx)

    if (len(best_indexes) == 0):
        best_action = random.choice(list(available_moves))
    else:
        best_idx = random.choice(best_indexes)
        best_action = list(available_moves)[best_idx]

    return best_action