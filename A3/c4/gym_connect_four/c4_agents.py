import random
import numpy as np
import c4.gym_connect_four.envs.connect_four_env as gym
import c4.gym_connect_four.c4_algorithms as algos
import collections


#! Random Agent
# Chooses an action of the available actions randomly
class RandomPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, name='RandomPlayer'):
        self.env = env
        self.player = player
        self.name = name

    def get_next_action(self):
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError(
                'Unable to determine a valid move! Maybe invoke at the wrong time?')

        action = random.choice(list(available_moves))
        return action


#! Clever Agent
# Picks a winning or blocking action, if possible, else picks a random action
class CleverPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, name='CleverPlayer'):
        self.env = env
        self.player = player
        self.name = name

    def get_next_action(self):
        available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError(
                'Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Check if player can win
        for action in list(available_moves):
            board = self.env.ghost_step(self.env.board, action, self.player)

            if self.env.ghost_is_terminal_state(board):
                if self.env.ghost_check_winner(board) == self.player:
                    return action

            del board

        # Check if opponent can win
        for action in available_moves:
            board = self.env.ghost_step(self.env.board, action, self.player*-1)

            if self.env.ghost_is_terminal_state(board):
                if self.env.ghost_check_winner(board) == self.player*-1:
                    return action

            del board

        return random.choice(list(available_moves))
        

#! Minimax Agent
# Chooses an action using Minimax algorithm
class MinimaxPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, max_player, name='MinimaxPlayer'):
        self.env = env
        self.player = player
        self.max_player = max_player
        self.name = name
        self.move_count = 0

    def get_next_action(self):
        depth = 0
        action = algos.minimax(self.env, self.env.board, [self.player, self.max_player], self.player, self.max_player, depth, self)
        return action[0]


#! Minimax Prune Agent
# Chooses an action using Minimax algorithm with alpha-beta pruning
class MinimaxPrunePlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, max_player, name='MinimaxPrunePlayer'):
        self.env = env
        self.player = player
        self.max_player = max_player
        self.name = name
        self.move_count = 0

    def get_next_action(self):
        depth = 0
        action = algos.minimax_prune(self.env, self.env.board, [self.player, self.max_player], self.player, self.max_player, depth, -999, 999, self)
        return action[0]


#! Minimax Heuristic Agent
# Chooses an action using Minimax algorithm and a heuristic
class MinimaxHeuristicPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, max_player, name='MinimaxHeuristicPlayer'):
        self.env = env
        self.player = player
        self.max_player = max_player
        self.name = name
        self.move_count = 0

    def get_next_action(self):
        depth = 0
        action = algos.minimax_heuristic(self.env, self.env.board, [self.player, self.max_player], self.player, self.max_player, depth, self)
        return action[0]


#! Minimax Prune Heuristic Agent
# Chooses an action using Minimax algorithm with alpha-beta pruning and a heuristic
class MinimaxPruneHeuristicPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, max_player, name='MinimaxPruneHeuristicPlayer'):
        self.env = env
        self.player = player
        self.max_player = max_player
        self.name = name
        self.move_count = 0

    def get_next_action(self):
        depth = 0
        action = algos.minimax_prune_heuristic(self.env, self.env.board, [self.player, self.max_player], self.player, self.max_player, depth, -999, 999, self)
        return action[0]


#! Qlearn Agent
# Chooses an action using Qlearning and Qtable
class QlearnPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, name='QlearnPlayer'):
        self.env = env
        self.player = player
        self.name = name
        self.qtable = {}
        self.epsilon = 1

        for action in range(0, 7):
            self.qtable[action] = collections.defaultdict(int)

    def get_next_action(self, training=False):
        if training:
            action, self.epsilon = algos.qlearnAct(self.env, self.qtable, self.epsilon)
        else:
            action, self.epsilon = algos.qlearnAct(self.env, self.qtable, 0)

        return action

    def update(self, prev_board, next_board, transition_action, score):
        algos.qlearnUpdate(self.env, self.qtable, prev_board, next_board, transition_action, score)
