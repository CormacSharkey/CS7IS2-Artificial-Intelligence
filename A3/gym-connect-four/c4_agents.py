import random
import numpy as np
# from abc import ABC, abstractmethod
import gym_connect_four.envs.connect_four_env as gym
import c4_algorithms as algos
import collections


# class Player(ABC):
#     """ Class used for evaluating the game """

#     def __init__(self, env: gym.ConnectFourEnv, name='Player'):
#         self.name = name
#         self.env = env

#     @abstractmethod
#     def get_next_action(self, state: np.ndarray) -> int:
#         pass

#     def learn(self, state, action: int, state_next, reward: int, done: bool) -> None:
#         pass

#     def save_model(self, model_prefix: str = None):
#         raise NotImplementedError()

#     def load_model(self, model_prefix: str = None):
#         raise NotImplementedError()

#     def reset(self, episode: int = 0, side: int = 1) -> None:
#         """
#         Allows a player class to reset it's state before each round

#             Parameters
#             ----------
#             episode : which episode we have reached
#             side : 1 if the player is starting or -1 if the player is second
#         """
#         pass

#! Random Agent
class RandomPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, name='RandomPlayer'):
        self.env = env
        self.player = player
        self.name = name
        
    def get_next_action(self):
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        action = random.choice(list(available_moves))
        return action
    
#! Clever Agent
class CleverPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, name='CleverPlayer'):
        self.env = env
        self.player = player
        self.name = name
        
    def get_next_action(self):
        available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Check if player can win
        for action in list(available_moves):
            board = self.env.ghost_step(self.env.board, action, self.player)

            if self.env.ghost_is_terminal_state(board):
                if self.env.ghost_check_winner(board) == self.player:
                    return action
            
            del board
            
            
        for action in available_moves:
            board = self.env.ghost_step(self.env.board, action, self.player*-1)

            if self.env.ghost_is_terminal_state(board):
                if self.env.ghost_check_winner(board) == self.player*-1:
                    return action
                
            del board
                
        print("Random Action!")
        action = random.choice(list(available_moves))
        return action

#! Minimax Agent
class MinimaxPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, maxPlayer, name='MinimaxPlayer'):
        self.env = env
        self.player = player
        self.maxPlayer = maxPlayer
        self.name = name
        
    def get_next_action(self):
        depth = 0
        action = algos.minimax(self.env, self.env.board, [self.player, self.maxPlayer], self.player, self.maxPlayer, depth)

        return action[0]
    
#! Minimax Prune Agent
class MinimaxPrunePlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, maxPlayer, name='MinimaxPrunePlayer'):
        self.env = env
        self.player = player
        self.maxPlayer = maxPlayer
        self.name = name
        
    def get_next_action(self):
        depth = 0
        action = algos.minimax_prune(self.env, self.env.board, [self.player, self.maxPlayer], self.player, self.maxPlayer, depth, -999, 999)

        return action[0]
    
#! Minimax Heuristic Agent
class MinimaxHeuristicPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, maxPlayer, name='MinimaxHeuristicPlayer'):
        self.env = env
        self.player = player
        self.maxPlayer = maxPlayer
        self.name = name
        
    def get_next_action(self):
        depth = 0
        action = algos.minimax_heuristic(self.env, self.env.board, [self.player, self.maxPlayer], self.player, self.maxPlayer, depth)

        return action[0]
    
#! Minimax Prune Heuristic Agent
class MinimaxPruneHeuristicPlayer():
    def __init__(self, env: gym.ConnectFourEnv, player, maxPlayer, name='MinimaxPruneHeuristicPlayer'):
        self.env = env
        self.player = player
        self.maxPlayer = maxPlayer
        self.name = name
        
    def get_next_action(self):
        depth = 0
        action = algos.minimax_prune_heuristic(self.env, self.env.board, [self.player, self.maxPlayer], self.player, self.maxPlayer, depth, -999, 999)

        return action[0]
    
#! Qlearn Agent
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
            action, self.qtable, self.epsilon = algos.qlearnAct(self.env, self.qtable, self.epsilon)
        else:
            action, self.qtable, self.epsilon = algos.qlearnAct(self.env, self.qtable, 0)

        return action
    
    def update(self, prev_board, next_board, transition_action, score):
        algos.qlearnUpdate(self.env, self.qtable, prev_board, next_board, transition_action, score)
    