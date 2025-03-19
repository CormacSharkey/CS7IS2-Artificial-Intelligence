import random
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status, after_action_state, tomark, next_mark
import algorithms as algos

#! Random Agent
# Picks a random action that is valid
class RandomAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "RA"

    def act(self, state, ava_actions):
        # Return a random action
        return random.choice(ava_actions)


#! Clever Agent
# Picks single-move winning actions and single-move blocking actions, else random actions
class CleverAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.indicator = "CA"

    def act(self, state, ava_actions):
        # For every action in the provided available actions
        for action in ava_actions:
            # Get the proposed state of the board after the ally takes the given action
            nstate_ally = after_action_state(state, action)
            # Get the game status (victory X, victory O, no-win, still playing) based on the proposed state
            gstatus_ally = check_game_status(nstate_ally[0])

            # If the game status is a victory
            if gstatus_ally > 0:
                # If the ally's mark is the victory mark
                if tomark(gstatus_ally) == self.mark:
                    # Return the current action (ensures victory for the agent)
                    return action

        for action in ava_actions:
            # Get the proposed state of the board after the enemy takes the given action
            nstate_enemy = after_action_state(
                (state[0], next_mark(state[1])), action)
            # Get the game status (victory X, victory O, no-win, still playing) based on the proposed state
            gstatus_enemy = check_game_status(nstate_enemy[0])

            # If the game status is a victory
            if gstatus_enemy > 0:
                # If the enemy's mark is the victory mark
                if tomark(gstatus_enemy) == next_mark(state[1]):
                    # Return the current action (ensures the enemy is blocked)
                    return action

        # If none of the available actions mean ally victory or enemy blocked, return a random action
        return random.choice(ava_actions)


#! Minimax Agent
# Applies the Minimax algorithm, using an indicator of max or min
class MinimaxAgent(object):
    def __init__(self, mark, player):
        self.mark = mark
        self.player = player
        self.indicator = "MA"

    def act(self, state):
        score_action = algos.minimax(state, self.player)
        print(score_action[0])
        return score_action[0]
    

def play(max_episode=1):
    # Set the starting mark (agent who goes first)
    start_mark = 'X'
    # Create the environment
    env = TicTacToeEnv()
    # Establish the two agents (O and X)
    agents = [CleverAgent('O'), RandomAgent('X')]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set the starting mark for the environment
        env.set_start_mark(start_mark)
        # Reset the environment to get the initial state
        state = env.reset()

        # While the game has not be won
        while not env.done:
            # Get the mark who moves next
            _, mark = state
            # Show which mark is currently acting
            env.show_turn(True, mark)

            # Get the agent who is acting now
            agent = agent_by_mark(agents, mark)

            if (agent.indicator == "MA"):

                action = agent.act(state)
            else:
                # Get all available actions (actions where there is an open spot on the board)
                ava_actions = env.available_actions()


                # Get the agent's action by giving it the available actions
                action = agent.act(state, ava_actions)

            # Update the environment with the agent's action
            state, reward, done, info = env.step(action)
            # Render the environment for viewing
            env.render()

        # Show the final results of the game; which mark won
        env.show_result(True, mark, reward)

        # Swap the start mark around for fairness (X, O, X, O, etc.)
        start_mark = next_mark(start_mark)


if __name__ == '__main__':
    play()
