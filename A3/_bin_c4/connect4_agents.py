import gym_connect4.envs.connect4_env as connect4
import random
import connect4_algorithms as algos

#! Clever Agent
class CleverAgent(object):
    def __init__(self, player):
        self.player = player

    def act(self, board: connect4.Connect4):
        ava_actions = board.get_moves()

        # For every action in the provided available actions
        for action in ava_actions:
            clone_board = board.clone()

            clone_board.move(action)
            if (clone_board.is_winner()):
                del clone_board
                return action
        
        enemy_board = board.clone()
        enemy_board.player ^= 1

        for action in ava_actions:
            clone_board = enemy_board.clone()

            clone_board.move(action)
            if (clone_board.is_winner()):
                del clone_board
                del enemy_board
                return action
            
        action = random.choice(ava_actions)
        return action

#! Random Agent
class RandomAgent(object):
    def __init__(self, player):
        self.player = player

    def act(self, board: connect4.Connect4):
        ava_actions = board.get_moves()

        # Return a random action
        return random.choice(ava_actions)
    

#! Minimax Raw Agent
# Applies the Minimax algorithm, using an indicator of max or min
class MinimaxRawAgent(object):
    def __init__(self, player, maxPlayer):
        self.player = player
        self.maxPlayer = maxPlayer

    def act(self, board):
        score_action = algos.minimax_raw(board, self.maxPlayer, [self.player, self.maxPlayer], 0)
        return score_action[0]
    

#! Minimax Heuristic Agent
# Applies the Minimax algorithm, using an indicator of max or min
class MinimaxHeuristicAgent(object):
    def __init__(self, player, maxPlayer):
        self.player = player
        self.maxPlayer = maxPlayer

    def act(self, board):
        score_action = algos.minimax_heuristic(board, self.maxPlayer, [self.player, self.maxPlayer], 0)
        return score_action[0]
