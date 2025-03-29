import random
import numpy as np
# from abc import ABC, abstractmethod
import gym_connect_four.envs.connect_four_env as gym



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

            if self.env.ghost_is_win_state(board):
                return action
            
            
        for action in available_moves:
            board = self.env.ghost_step(self.env.board, action, self.player*-1)

            if self.env.ghost_is_win_state(board):
                return action
            
        print("Random Action!")
        action = random.choice(list(available_moves))
        return action
    