from ttt.gym_tictactoe import env as ttt_env
from ttt.gym_tictactoe import ttt_agents as ttt_agents
from ttt.gym_tictactoe import ttt_algorithms as ttt_algos

from c4.gym_connect_four.envs import connect_four_env as c4_env
from c4.gym_connect_four import c4_agents as c4_agents

import gym
import argparse
import time


def ttt_play(agents, max_episode=1, render=True):
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    p1_matches = [0, 0, 0]
    p2_matches = [0, 0, 0]

    p1_actions = 0

    for _ in range(max_episode):
        env.set_start_mark(start_mark)
        state = env.reset()

        while not env.done:
            _, mark = state
            if render:
                env.show_turn(True, agents, mark)

            if mark == 'O':
                p1_actions += 1

            agent = ttt_env.agent_by_mark(agents, mark)

            if (agent.indicator == "RA" or agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            elif (agent.indicator == "CA"):
                action = agent.act(state, env.available_actions())
            else:
                action = agent.act(state)

            state, reward, done, info = env.step(action)
            if render:
                env.render()

            if (env.done):
                break

            _, mark = state
            if render:
                env.show_turn(True, agents, mark)

            agent = ttt_env.agent_by_mark(agents, mark)

            if (agent.indicator == "RA" or agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            elif (agent.indicator == "CA"):
                action = agent.act(state, env.available_actions())
            else:
                action = agent.act(state)

            state, reward, done, info = env.step(action)
            if render:
                env.render()
        
        result = env.show_result(True, agents, reward, render)

        if start_mark == 'O':
            if result == 1:
                p1_matches[0] += 1
            elif result == 2:
                p1_matches[1] += 1
            elif result == 0:
                p1_matches[2] += 1
        else:
            if result == 1:
                p2_matches[1] += 1
            elif result == 2:
                p2_matches[0] += 1
            elif result == 0:
                p2_matches[2] += 1

        start_mark = ttt_env.next_mark(start_mark)

    if agents[0].indicator == "MA":
        return [p1_matches, p2_matches, p1_actions/max_episode, agents[0].move_count/max_episode]
    else:
        return [p1_matches, p2_matches, p1_actions/max_episode]


def ttt_train(max_episode=1, lr = 0.5, discount = 0.9):
    start_time = time.time()

    # Set starting mark and create env
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    agents = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]

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

            ttt_algos.qlearnUpdate(agents[0].qtable, prev_state, state, transition_action, score, lr, discount)

            prev_state = state
            transition_action = new_action

        ttt_algos.qlearnUpdate(agents[0].qtable, prev_state, None, transition_action, score)

        # Show the game's result and swap the start mark for the next episode
        start_mark = ttt_env.next_mark(start_mark)

    finish_time = time.time()
    time_to_train = finish_time - start_time

    return agents[0].qtable, time_to_train


def ttt_flow(opt):
    match opt.players:
        case "MC":
            playerMC = [ttt_agents.MinimaxAgent('O', True), ttt_agents.CleverAgent('X')]
            metrics = ttt_play(playerMC, opt.episodes)
            return metrics
        case "MPC":
            playerMPC = [ttt_agents.MinimaxPruneAgent('O', True), ttt_agents.CleverAgent('X')]
            metrics = ttt_play(playerMPC, opt.episodes)
            return metrics
        case "QC":
            playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
            playersQC[0].qtable, time = ttt_train(opt.ttt_epochs)
            metrics = ttt_play(playersQC, opt.episodes)
            metrics.append(time)
            return metrics
        case "QM":
            playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxAgent('X', True)]
            playersQM[0].qtable, time = ttt_train(opt.ttt_epochs)
            metrics = ttt_play(playersQM, opt.episodes)
            metrics.append(time)
            return metrics
        case _:
            print("Error: No matching players")


def c4_play(env: c4_env.ConnectFourEnv, players, max_episode=1, render=True):
    start_player = 1

    p1_matches = [0, 0, 0]
    p2_matches = [0, 0, 0]

    p1_actions = 0

    for _ in range(max_episode):
        result, p1 = env.run_game(players[0], players[1], env.board, render, start_player)
        p1_actions += p1
        
        if start_player == 1:
            if result == c4_env.ResultType.WIN1:
                p1_matches[0] += 1
            elif result == c4_env.ResultType.WIN2:
                p1_matches[1] += 1
            elif result == c4_env.ResultType.DRAW:
                p1_matches[2] += 1
        else:
            if result == c4_env.ResultType.WIN1:
                p2_matches[1] += 1
            elif result == c4_env.ResultType.WIN2:
                p2_matches[0] += 1
            elif result == c4_env.ResultType.DRAW:
                p2_matches[2] += 1

        start_player *= -1
    
    if players[0].name[0:7] == "Minimax":
        return [p1_matches, p2_matches, p1_actions/max_episode, players[0].move_count/max_episode]
    else:
        return [p1_matches, p2_matches, p1_actions/max_episode]


def c4_train(env: c4_env.ConnectFourEnv, max_episode=1):
    start_time = time.time()
    player = c4_agents.QlearnPlayer(env, 1, 'QlearnPlayer')
    opponent = c4_agents.CleverPlayer(env, -1, 'OpponentCleverPlayer')

    players = [player, opponent]

    start_player = 1

    for _ in range(max_episode):
        result = env.train_game(players[0], players[1], env.board, True, start_player)
        start_player *= -1

    finish_time = time.time()
    time_to_train = finish_time - start_time

    return players[0].qtable, time_to_train


def c4_flow(opt):
    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    match opt.players:
        case "MC":
            playerMC = [c4_agents.MinimaxHeuristicPlayer(env, 1, True, "MinimaxHeuristicPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
            metrics = c4_play(env, playerMC, opt.episodes)
            return metrics
        case "MPC":
            playerMPC = [c4_agents.MinimaxPruneHeuristicPlayer(env, 1, True, "MinimaxPruneHeuristicPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
            metrics = c4_play(env, playerMPC, opt.episodes)
            return metrics
        case "QC":
            playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
            playersQC[0].qtable, time = c4_train(env, opt.c4_epochs)
            metrics = c4_play(env, playersQC, opt.episodes)
            metrics.append(time)
            return metrics
        case "QM":
            playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxHeuristicPlayer(env, -1, True, "OpponentMinimaxHeuristicPlayer")]
            playersQM[0].qtable, time = c4_train(env, opt.c4_epochs)
            metrics = c4_play(env, playersQM, opt.episodes)
            metrics.append(time)
            return metrics
        case _:
            print("Error: No matching players")


def all_flow(opt):
    print("Running ttt: MC")
    playerMC = [ttt_agents.MinimaxAgent('O', True), ttt_agents.CleverAgent('X')]
    metrics = ttt_play(playerMC, opt.episodes)

    print("Running ttt: MPC")
    playerMPC = [ttt_agents.MinimaxPruneAgent('O', True), ttt_agents.CleverAgent('X')]
    metrics = ttt_play(playerMPC, opt.episodes)

    print("Running ttt: QC")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable = ttt_train(opt.ttt_epochs)
    metrics = ttt_play(playersQC, opt.episodes)

    print("Running ttt: QM")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxAgent('X', True)]
    playersQM[0].qtable = ttt_train(opt.ttt_epochs)
    metrics = ttt_play(playersQM, opt.episodes)


    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")

    print("Running c4: MC")
    playerMC = [c4_agents.MinimaxHeuristicPlayer(env, 1, True, "MinimaxHeuristicPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    metrics = c4_play(env, playerMC, opt.episodes)

    print("Running c4: MPC")
    playerMPC = [c4_agents.MinimaxPruneHeuristicPlayer(env, 1, True, "MinimaxPruneHeuristicPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    metrics = c4_play(env, playerMPC, opt.episodes)

    print("Running c4: QC")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable = c4_train(env, opt.c4_epochs)
    metrics = c4_play(env, playersQC, opt.episodes)

    print("Running c4: QM")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxHeuristicPlayer(env, -1, True, "OpponentMinimaxHeuristicPlayer")]
    playersQM[0].qtable = c4_train(env, opt.c4_epochs)
    metrics = c4_play(env, playersQM, opt.episodes)


def test_bench_flow():
    # Test_bench:

    episodes = 100 
    
    ###################################################################################################
    """
    #* Run TTT MC for 100 games, get results
    print("Running ttt: MC")
    playerMC = [ttt_agents.MinimaxAgent('O', True), ttt_agents.CleverAgent('X')]
    metrics = ttt_play(playerMC, episodes, False)
    print(f"ttt MC metrics: {metrics}")

    #* Run TTT MPC for 100 games, get results
    print("Running ttt: MPC")
    playerMPC = [ttt_agents.MinimaxPruneAgent('O', True), ttt_agents.CleverAgent('X')]
    metrics = ttt_play(playerMPC, episodes, False)
    print(f"ttt MPC metrics: {metrics}")

    ###################################################################################################
    #* Run TTT QC, vary training length of 1000, 5000, 10000, 25000, 50000
    print("Running ttt: QC 1000")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(1000)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 1000 metrics: {metrics}")

    print("Running ttt: QC 5000")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(5000)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 5000 metrics: {metrics}")

    print("Running ttt: QC 10000")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(10000)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 10000 metrics: {metrics}")

    print("Running ttt: QC 25000")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(25000)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 25000 metrics: {metrics}")

    print("Running ttt: QC 50000")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(50000)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 50000 metrics: {metrics}")

    ###################################################################################################

    #* Varying parameters
    print("Running ttt: QC 50000 lr=0.75")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(50000, lr=0.75)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 50000 lr=0.75 metrics: {metrics}")

    print("Running ttt: QC 50000 lr = 0.25")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(50000, lr = 0.25)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 50000 lr = 0.25 metrics: {metrics}")

    print("Running ttt: QC 50000 discount = 0.5")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(50000, discount=0.5)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 50000 discount = 0.5 metrics: {metrics}")

    print("Running ttt: QC 50000 discount = 0.1")
    playersQC = [ttt_agents.QLearningAgent('O'), ttt_agents.CleverAgent('X')]
    playersQC[0].qtable, time = ttt_train(50000, discount = 0.1)
    metrics = ttt_play(playersQC, episodes, False)
    metrics.append(time)
    print(f"ttt QC 50000 lr = 0.1 metrics: {metrics}")

    ###################################################################################################

    #* Run TTT QM, vary training length of 1000, 5000, 10000, 25000, 50000
    print("Running ttt: QM 1000")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(1000)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 1000 metrics: {metrics}")

    print("Running ttt: QM 5000")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(5000)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 5000 metrics: {metrics}")

    print("Running ttt: QM 10000")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(10000)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 10000 metrics: {metrics}")

    print("Running ttt: QM 25000")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(25000)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 25000 metrics: {metrics}")

    print("Running ttt: QM 50000")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(50000)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 50000 metrics: {metrics}")

    ###################################################################################################

    print("Running ttt: QM 50000 lr = 0.25")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(50000, lr = 0.25)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 50000 lr = 0.25 metrics: {metrics}")

    print("Running ttt: QM 50000 lr = 0.75")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(50000, lr = 0.75)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 50000 lr = 0.75 metrics: {metrics}")

    print("Running ttt: QM 50000 discount = 0.5")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(50000, discount = 0.5)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 50000 discount = 0.5 metrics: {metrics}")

    print("Running ttt: QM 50000 discount = 0.1")
    playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxPruneAgent('X', True)]
    playersQM[0].qtable, time = ttt_train(50000, discount = 0.1)
    metrics = ttt_play(playersQM, episodes, False)
    metrics.append(time)
    print(f"ttt QM 50000 discount = 0.1 metrics: {metrics}")
    """
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    # #* Run C4 MC no heuristic for 30 mins, determine the number of states it saw
    # env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    # print("Running c4: MC")
    # playerMC = [c4_agents.MinimaxPlayer(env, 1, True, "MinimaxPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    # metrics = c4_play(env, playerMC, episodes, False)
    # print(f"c4 MC metrics: {metrics}")

    # #* Run C4 MPC no heuristic for 30 mins, determine the number of states it saw
    # env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    # print("Running c4: MC")
    # playerMC = [c4_agents.MinimaxPrunePlayer(env, 1, True, "MinimaxPrunePlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    # metrics = c4_play(env, playerMC, episodes, False)
    # print(f"c4 MC metrics: {metrics}")

    ###################################################################################################
    """
    #* Run C4 MC for 100 games, get results
    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    print("Running c4: MHC")
    playerMC = [c4_agents.MinimaxHeuristicPlayer(env, 1, True, "MinimaxHeuristicPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    metrics = c4_play(env, playerMC, episodes, False)
    print(f"c4 MHC metrics: {metrics}")

    #* Run C4 MPC for 1000 games, get results
    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    print("Running c4: MPHC")
    playerMPC = [c4_agents.MinimaxPruneHeuristicPlayer(env, 1, True, "MinimaxPruneHeuristicPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    metrics = c4_play(env, playerMPC, episodes, False)
    print(f"c4 MPHC metrics: {metrics}")

    ###################################################################################################
    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    print("Running c4: QC 50000 full")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable, time = c4_train(env, 50000)
    metrics = c4_play(env, playersQC, episodes, False)
    metrics.append(time)
    print(f"c4 QC metrics 50000 full: {metrics}")

    ###################################################################################################

    #* Run QC reduced board, vary training length of 100, 1000, 10000, 50000, for 1000 games, get results
    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(4,5))
    print("Running c4: QC 1000 reduced")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable, time = c4_train(env, 1000)
    metrics = c4_play(env, playersQC, episodes, False)
    metrics.append(time)
    print(f"c4 QC metrics 1000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(4,5))
    print("Running c4: QC 5000 reduced")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable, time = c4_train(env, 5000)
    metrics = c4_play(env, playersQC, episodes, False)
    metrics.append(time)
    print(f"c4 QC metrics 5000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(4,5))
    print("Running c4: QC 10000 reduced")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable, time = c4_train(env, 10000)
    metrics = c4_play(env, playersQC, episodes, False)
    metrics.append(time)
    print(f"c4 QC metrics 10000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(4,5))
    print("Running c4: QC 25000 reduced")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable, time = c4_train(env, 25000)
    metrics = c4_play(env, playersQC, episodes, False)
    metrics.append(time)
    print(f"c4 QC metrics 25000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(4,5))
    print("Running c4: QC 50000 reduced")
    playersQC = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.CleverPlayer(env, -1, "OpponentCleverPlayer")]
    playersQC[0].qtable, time = c4_train(env, 50000)
    metrics = c4_play(env, playersQC, episodes, False)
    metrics.append(time)
    print(f"c4 QC metrics 50000 reduced: {metrics}")

    ###################################################################################################

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0")
    print("Running c4: QMPH 50000 full")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxPruneHeuristicPlayer(env, -1, True, "OpponentMinimaxPruneHeuristicPlayer")]
    playersQM[0].qtable, time = c4_train(env, 50000)
    metrics = c4_play(env, playersQM, episodes, False)
    metrics.append(time)
    print(f"c4 QMPH metrics 50000 full: {metrics}")

    ###################################################################################################

    #* Run QMPH reduced board, vary training length of 1000, 10000, 50000, for 1000 games
    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(5,7))
    print("Running c4: QMPH 1000 reduced")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxPruneHeuristicPlayer(env, -1, True, "OpponentMinimaxPruneHeuristicPlayer")]
    playersQM[0].qtable, time = c4_train(env, 1000)
    metrics = c4_play(env, playersQM, episodes, False)
    metrics.append(time)
    print(f"c4 QMPH metrics 1000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(5,7))
    print("Running c4: QMPH 5000 reduced")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxPruneHeuristicPlayer(env, -1, True, "OpponentMinimaxPruneHeuristicPlayer")]
    playersQM[0].qtable, time = c4_train(env, 5000)
    metrics = c4_play(env, playersQM, episodes, False)
    metrics.append(time)
    print(f"c4 QMPH metrics 5000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(5,7))
    print("Running c4: QMPH 10000 reduced")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxPruneHeuristicPlayer(env, -1, True, "OpponentMinimaxPruneHeuristicPlayer")]
    playersQM[0].qtable, time = c4_train(env, 10000)
    metrics = c4_play(env, playersQM, episodes, False)
    metrics.append(time)
    print(f"c4 QMPH metrics 10000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(5,7))
    print("Running c4: QMPH 25000 reduced")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxPruneHeuristicPlayer(env, -1, True, "OpponentMinimaxPruneHeuristicPlayer")]
    playersQM[0].qtable, time = c4_train(env, 25000)
    metrics = c4_play(env, playersQM, episodes, False)
    metrics.append(time)
    print(f"c4 QMPH metrics 25000 reduced: {metrics}")

    env: c4_env.ConnectFourEnv = gym.make("ConnectFour-v0", board_shape=(5,7))
    print("Running c4: QMPH 50000 reduced")
    playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxPruneHeuristicPlayer(env, -1, True, "OpponentMinimaxPruneHeuristicPlayer")]
    playersQM[0].qtable, time = c4_train(env, 50000)
    metrics = c4_play(env, playersQM, episodes, False)
    metrics.append(time)
    print(f"c4 QMPH metrics 50000 reduced: {metrics}")

    ###################################################################################################

    """


if __name__ == '__main__':
    # Parse the command line args
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='ttt', help='Which game to play?')
    parser.add_argument('--players', type=str, default='MPC', help='Which agents are playing?')
    parser.add_argument('--episodes', type=int, default=1, help='How many game rounds?')
    parser.add_argument('--ttt_epochs', type=int, default=50000, help='How long will ttt train?')
    parser.add_argument('--c4_epochs', type=int, default=10000, help='How long will c4 train?')
    args = parser.parse_args()

    if args.game == "ttt":
        metrics = ttt_flow(args)
        print(metrics)
    elif args.game == "c4":
        metrics = c4_flow(args)
        print(metrics)
    elif args.game == "all":
        metrics = all_flow(args)
        print(metrics)
    elif args.game == "test":
        test_bench_flow()
    else:
        print("Error: No matching game")
        
