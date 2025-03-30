import gym_tictactoe.env as ttt_env
from gym_tictactoe.ttt_agents import RandomAgent, CleverAgent, HumanAgent, MinimaxAgent, MinimaxPruneAgent, QLearningAgent
import gym_tictactoe.ttt_algorithms as algos


#! Play
def play(agents, max_episode=1):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, agents, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            if (agent.indicator == "RA" or agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            elif (agent.indicator == "CA"):
                action = agent.act(state, env.available_actions())
            else:
                action = agent.act(state)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

            # If the game is won, end it
            if (env.done):
                break

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, agents, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            if (agent.indicator == "RA" or agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            elif (agent.indicator == "CA"):
                action = agent.act(state, env.available_actions())
            else:
                action = agent.act(state)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()
        

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, agents, reward)
        start_mark = ttt_env.next_mark(start_mark)


#! Train Qlearn Agent
def trainQlearnAgent(max_episode=1):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    agents = [QLearningAgent('O'), CleverAgent('X')]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        if (ttt_env.agent_by_mark(agents, start_mark).indicator != "QA"):
            first_state = state
            _, mark = first_state

            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)

        prev_state = state
        _, mark = prev_state
        transition_action = ttt_env.agent_by_mark(agents, mark).act(prev_state, True)

        score = 0

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = prev_state

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(transition_action)

            # If the game is won, end it
            if (env.done):
                if ttt_env.check_game_status(state[0]) == ttt_env.tocode(mark):
                    score = 1
                    break
                else:
                    score = 0
                    break

            # Get current turn's mark and show the turn
            _, mark = state

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)

            # If the game is won, end it
            if (env.done):
                if ttt_env.check_game_status(state[0]) == ttt_env.tocode(mark):
                    score = -1
                    break
                else:
                    score = 0
                    break

            _, mark = state
            agent = ttt_env.agent_by_mark(agents, mark)
            new_action = agent.act(state, True)

            algos.qlearnUpdate(agents[0].qtable, prev_state, state, transition_action, score)

            prev_state = state
            transition_action = new_action

        algos.qlearnUpdate(agents[0].qtable, prev_state, None, transition_action, score)

        # Show the game's result and swap the start mark for the next episode
        start_mark = ttt_env.next_mark(start_mark)


    return agents[0].qtable


if __name__ == '__main__':
    agents = [QLearningAgent('O'), CleverAgent('X')]

    qtable = trainQlearnAgent(50000)

    print("Training Complete!")

    agents[0].qtable = qtable
    play(agents, 5)
