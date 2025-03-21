import random
import gym_tictactoe.env as ttt_env
import algorithms as algos
import utils

#! Random Agent
# Agent that behaves by selecting an available action randomly
class RandomAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "RA"

    def act(self, state, ava_actions):
        # Return a random action from the available actions
        return random.choice(ava_actions)


#! Clever Agent
# Agent that always picks a winning or blocking action if possible, else selects an available action randomly
class CleverAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "CA"

    def act(self, state, ava_actions):
        # For every every available action
        for action in ava_actions:
            # Get the proposed state of the board after the ally agent takes the given action
            nstate_ally = ttt_env.after_action_state(state, action)
            # Get the game status (victory X, victory O, no-win, still playing) based on the proposed state
            gstatus_ally = ttt_env.check_game_status(nstate_ally[0])

            # If the game status is a victory
            if gstatus_ally > 0:
                # If the ally agent's mark is the victory mark
                if ttt_env.tomark(gstatus_ally) == self.mark:
                    # Return the current action (ensures victory for the ally agent)
                    return action

        # For every every available action
        for action in ava_actions:
            # Get the proposed state of the board after the enemy agent takes the given action
            nstate_enemy = ttt_env.after_action_state((state[0], ttt_env.next_mark(state[1])), action)
            # Get the game status (victory X, victory O, no-win, still playing) based on the proposed state
            gstatus_enemy = ttt_env.check_game_status(nstate_enemy[0])

            # If the game status is a victory
            if gstatus_enemy > 0:
                # If the enemy agent's mark is the victory mark
                if ttt_env.tomark(gstatus_enemy) == ttt_env.next_mark(state[1]):
                    # Return the current action (ensures the enemy agent is blocked)
                    return action

        # If none of the available actions mean ally agent victory or enemy agent blocked, return a random available action
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
    def __init__(self, mark, maxPlayer):
        self.mark = mark
        self.maxPlayer = maxPlayer
        self.indicator = "MA"

    def act(self, state):
        # Might need to set depth as 9-the remaining moves to make
        depth = len(utils.get_valid_actions(state[0]))
        state = (state[0], state[1], self.maxPlayer)
        score_action = algos.minimax(state, self.maxPlayer, self.mark, depth)
        return score_action[0]
    
#! Minimax Prune Agent
# Applies the Minimax algorithm w/ Alpha-Beta Pruning, using an indicator of max or min
class MinimaxPruneAgent(object):
    def __init__(self, mark, maxPlayer):
        self.mark = mark
        self.maxPlayer = maxPlayer
        self.indicator = "MA"

    def act(self, state):
        # Might need to set depth as 9-the remaining moves to make
        depth = len(utils.get_valid_actions(state[0]))
        state = (state[0], state[1], self.maxPlayer)
        score_action = algos.minimax_alpha_beta_prune(state, self.maxPlayer, self.mark, depth, -999, 999)
        return score_action[0]