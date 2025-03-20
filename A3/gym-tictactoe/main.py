import gym_tictactoe.env as ttt_env
from agents import RandomAgent, CleverAgent, HumanAgent, MinimaxAgent


#! Play
# Game loop - sets up an environment and starts a gameplay loop between the two agents
def play(max_episode=1):
    # Set the starting mark (agent who goes first)
    start_mark = 'O'
    # Create the environment
    env = ttt_env.TicTacToeEnv()

    # Establish the two agents (O and X)
    # agents = [CleverAgent('O'), RandomAgent('X')]
    # agents = [MinimaxAgent('O', "max"), MinimaxAgent('X', "min")]
    # agents = [MinimaxAgent('O', "max"), CleverAgent("X")]
    # agents = [HumanAgent('O'), MinimaxAgent("X", "max")]
    agents = [MinimaxAgent('O', "max"), RandomAgent('X')]
    # agents = [HumanAgent('O'), CleverAgent("X")]

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
            agent = ttt_env.agent_by_mark(agents, mark)

            if (agent.indicator == "MA"):
                action = agent.act(state)

            elif (agent.indicator == "HA"):
                ava_actions = env.available_actions()
                action = agent.act(ava_actions)

            else:
                ava_actions = env.available_actions()
                action = agent.act(state, ava_actions)

            # Update the environment with the agent's action
            state, reward, done, info = env.step(action)
            # Render the environment for viewing
            env.render()

        # Show the final results of the game; which mark won
        env.show_result(True, mark, reward)

        # Swap the start mark around for fairness (X, O, X, O, etc.)
        start_mark = ttt_env.next_mark(start_mark)


if __name__ == '__main__':
    play()
