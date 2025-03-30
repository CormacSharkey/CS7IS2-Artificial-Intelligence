import random
import collections
import ttt.gym_tictactoe.env as ttt_env
import ttt. gym_tictactoe.ttt_algorithms as algos
import  ttt.gym_tictactoe.ttt_utils as ttt_utils


#! Random Agent
# Agent that behaves by selecting an available action randomly
class RandomAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "RA"

    def act(self, ava_actions):
        # Return a random action from the available actions
        return random.choice(ava_actions)


#! Clever Agent
# Agent that always picks a winning or blocking action if possible, else selects an available action randomly
class CleverAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "CA"

    def act(self, state, ava_actions):
        # For every available action (player loop)
        for action in ava_actions:
            # Get a future state based on the given action, then get the game status of the state (win, lose, draw)
            nstate_ally = ttt_env.after_action_state(state, action)
            gstatus_ally = ttt_env.check_game_status(nstate_ally[0])

            # If an agent has won, and this agent is the winner, return the action
            if gstatus_ally > 0:
                if ttt_env.tomark(gstatus_ally) == self.mark:
                    return action

        # For every available action (opponent loop)
        for action in ava_actions:
            # Get a future state based on the given action, then get the game status of the state (win, lose, draw)
            nstate_enemy = ttt_env.after_action_state((state[0], ttt_env.next_mark(state[1])), action)
            gstatus_enemy = ttt_env.check_game_status(nstate_enemy[0])

            # If an agent has won, and the opponent is the winner, return the action
            if gstatus_enemy > 0:
                if ttt_env.tomark(gstatus_enemy) == ttt_env.next_mark(state[1]):
                    return action

        # If no winning moves for player or agent are possible, pick randomly
        return random.choice(ava_actions)
    

#! Human Agent
# Agent that is controlled by the user, who is prompted to input an action when it is their turn
# Code here taken from original gym-tictactoe codebase
class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "HA"

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action


#! Minimax Agent
# Applies the Minimax algorithm, using an indicator of max or min
class MinimaxAgent(object):
    def __init__(self, mark, max_player):
        self.mark = mark
        self.max_player = max_player
        self.indicator = "MA"

    def act(self, state):
        state = (state[0], state[1], self.max_player)
        score_action = algos.minimax(state, self.max_player, self.mark)
        return score_action[0]
    
    
#! Minimax Prune Agent
# Applies the Minimax algorithm w/ Alpha-Beta Pruning, using an indicator of max or min
class MinimaxPruneAgent(object):
    def __init__(self, mark, max_player):
        self.mark = mark
        self.max_player = max_player
        self.indicator = "MA"

    def act(self, state):
        state = (state[0], state[1], self.max_player)
        score_action = algos.minimax_alpha_beta_prune(state, self.max_player, self.mark, -999, 999)
        return score_action[0]
    
#! Qlearning Agent
class QLearningAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "QA"
        self.qtable = {}
        self.epsilon = 1

        for action in range(0, 9):
            self.qtable[action] = collections.defaultdict(int)

    def act(self, state, training=False):
        if training:
            action, self.epsilon = algos.qlearnAct(state, self.qtable, self.epsilon)
        else:
            action, self.epsilon = algos.qlearnAct(state, self.qtable, 0)

        return action
        

