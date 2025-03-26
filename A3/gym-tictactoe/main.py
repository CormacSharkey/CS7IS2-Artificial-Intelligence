import gym_tictactoe.env as ttt_env
from agents import RandomAgent, CleverAgent, HumanAgent, MinimaxAgent, MinimaxPruneAgent, QLearningAgent
import algorithms as algos


def playMinimaxMinimax(max_episode=1, prune=False):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    if prune:
        agents = [MinimaxPruneAgent('O', True), MinimaxPruneAgent('X', False)]
    else:
        agents = [MinimaxAgent('O', True), MinimaxAgent('X', False)]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            action = agent.act(state)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

            # If the game is won, end it
            if env.done:
                break

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            action = agent.act(state)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()
        

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        start_mark = ttt_env.next_mark(start_mark)


def playMinimaxClever(max_episode=1):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    agents = [MinimaxAgent('O', "max"), CleverAgent("X")]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            if (agent.indicator == "MA"):
                action = agent.act(state)

            else:
                action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

            # If the game is won, end it
            if (env.done):
                break

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            if (agent.indicator == "MA"):
                action = agent.act(state)

            else:
                action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()
        

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        start_mark = ttt_env.next_mark(start_mark)


def playMinimaxRandom(max_episode=1):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    agents = [MinimaxAgent('O', "max"), RandomAgent('X')]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            action = agent.act(state)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

            # If the game is won, end it
            if (env.done):
                break

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()
        

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        start_mark = ttt_env.next_mark(start_mark)


def playQlearningClever(max_episode=1):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    agents = [QLearningAgent('O'), CleverAgent('X')]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # check if the first agent is qlearning
        # if not, move other agent and uodate state

        # set previous state and previous action for qlearning

        if (ttt_env.agent_by_mark(agents, start_mark).indicator != "QA"):
            _, mark = state
            env.show_turn(True, mark)

            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

        prev_state = state
        _, mark = prev_state
        transition_action = ttt_env.agent_by_mark(agents, mark).act(prev_state)

        score = 0

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = prev_state
            env.show_turn(True, mark)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(transition_action)
            env.render()

            # If the game is won, end it
            if (env.done):
                if ttt_env.check_game_status(state[0]) == ttt_env.tocode(mark):
                    score = 10
                    break
                else:
                    score = 0
                    break

            # algos.qlearnUpdate(ttt_env.agent_by_mark(agents, mark).qtable, prev_state, state, transition_action, score)

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

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
            new_action = agent.act(state)

            algos.qlearnUpdate(agent.qtable, prev_state, state, transition_action, score)

            prev_state = state
            transition_action = new_action

        if (agents[0].indicator == "QA"):    
            algos.qlearnUpdate(agents[0].qtable, prev_state, None, transition_action, score)
        else:
            algos.qlearnUpdate(agents[1].qtable, prev_state, None, transition_action, score)

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        # start_mark = ttt_env.next_mark(start_mark)


def playQlearningRandom(max_episode=1):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    agents = [QLearningAgent('O'), RandomAgent('X')]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # check if the first agent is qlearning
        # if not, move other agent and uodate state

        # set previous state and previous action for qlearning

        if (ttt_env.agent_by_mark(agents, start_mark).indicator != "QA"):
            _, mark = state
            env.show_turn(True, mark)

            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

        prev_state = state
        _, mark = prev_state
        transition_action = ttt_env.agent_by_mark(agents, mark).act(prev_state)

        score = 0

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = prev_state
            env.show_turn(True, mark)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(transition_action)
            env.render()

            # If the game is won, end it
            if (env.done):
                if ttt_env.check_game_status(state[0]) == ttt_env.tocode(mark):
                    score = 10
                    break
                else:
                    score = 0
                    break

            # algos.qlearnUpdate(ttt_env.agent_by_mark(agents, mark).qtable, prev_state, state, transition_action, score)

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

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
            new_action = agent.act(state)

            algos.qlearnUpdate(agent.qtable, prev_state, state, transition_action, score)

            prev_state = state
            transition_action = new_action

        if (agents[0].indicator == "QA"):    
            algos.qlearnUpdate(agents[0].qtable, prev_state, None, transition_action, score)
        else:
            algos.qlearnUpdate(agents[1].qtable, prev_state, None, transition_action, score)

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        # start_mark = ttt_env.next_mark(start_mark)


def playQlearningMinimax(max_episode=1, prune=False):
    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    if prune:
        agents = [QLearningAgent('O'), MinimaxPruneAgent('X', True)]
    else:
        agents = [QLearningAgent('O'), MinimaxAgent('X', True)]

    # For loop - loop through episodes for max_episodes duration
    for _ in range(max_episode):
        # Set starting mark in env and reset env to get state
        env.set_start_mark(start_mark)
        state = env.reset()

        # check if the first agent is qlearning
        # if not, move other agent and uodate state

        # set previous state and previous action for qlearning

        if (ttt_env.agent_by_mark(agents, start_mark).indicator != "QA"):
            _, mark = state
            env.show_turn(True, mark)

            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

        prev_state = state
        _, mark = prev_state
        transition_action = ttt_env.agent_by_mark(agents, mark).act(prev_state)

        score = 0

        # While the game has not be won
        while not env.done:
            # Get current turn's mark and show the turn
            _, mark = prev_state
            env.show_turn(True, mark)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(transition_action)
            env.render()

            # If the game is won, end it
            if (env.done):
                if ttt_env.check_game_status(state[0]) == ttt_env.tocode(mark):
                    score = 10
                    break
                else:
                    score = 0
                    break

            # algos.qlearnUpdate(ttt_env.agent_by_mark(agents, mark).qtable, prev_state, state, transition_action, score)

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            action = agent.act(state)

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

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
            new_action = agent.act(state)

            algos.qlearnUpdate(agent.qtable, prev_state, state, transition_action, score)

            prev_state = state
            transition_action = new_action

        if (agents[0].indicator == "QA"):    
            algos.qlearnUpdate(agents[0].qtable, prev_state, None, transition_action, score)
        else:
            algos.qlearnUpdate(agents[1].qtable, prev_state, None, transition_action, score)

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        # start_mark = ttt_env.next_mark(start_mark)


#! Play
# Game loop - sets up an environment and starts a gameplay loop between the two agents
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
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            if (agent.indicator == "MA"):
                action = agent.act(state)

            elif (agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            
            elif (agent.indicator == "QA"):
                action = agent.act(state)

            else:
                action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()

            # If the game is won, end it
            if (env.done):
                break

            # Get current turn's mark and show the turn
            _, mark = state
            env.show_turn(True, mark)

            # Get the agent whose turn it is
            agent = ttt_env.agent_by_mark(agents, mark)

            # If-Else block to determine which agent is going, and how to act
            if (agent.indicator == "MA"):
                action = agent.act(state)

            elif (agent.indicator == "HA"):
                action = agent.act(env.available_actions())

            elif (agent.indicator == "QA"):
                action = agent.act(state)

            else:
                action = agent.act(state, env.available_actions())

            # Get the state and reward for the agent's move, and render the maze for the user
            state, reward, done, info = env.step(action)
            env.render()
        

        # Show the game's result and swap the start mark for the next episode
        env.show_result(True, mark, reward)
        start_mark = ttt_env.next_mark(start_mark)


def main():
    # Establish the two agents (O and X)
    # playMinimaxMinimax(prune=False)

    # playMinimaxMinimax(prune=True)

    playQlearningClever(5000)
    # playQlearningRandom(5000)
    # playQlearningMinimax(5000, True)


if __name__ == '__main__':
    main()
