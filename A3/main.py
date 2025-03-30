from ttt.gym_tictactoe import env as ttt_env
from ttt.gym_tictactoe import ttt_agents as ttt_agents
from ttt.gym_tictactoe import ttt_algorithms as ttt_algos

from c4.gym_connect_four.envs import connect_four_env as c4_env
from c4.gym_connect_four import c4_agents as c4_agents

import gym
import argparse


def ttt_play(agents, max_episode=1):
    start_mark = 'O'
    env = ttt_env.TicTacToeEnv()

    for _ in range(max_episode):
        env.set_start_mark(start_mark)
        state = env.reset()

        while not env.done:
            _, mark = state
            env.show_turn(True, agents, mark)

            agent = ttt_env.agent_by_mark(agents, mark)

            if (agent.indicator == "RA" or agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            elif (agent.indicator == "CA"):
                action = agent.act(state, env.available_actions())
            else:
                action = agent.act(state)

            state, reward, done, info = env.step(action)
            env.render()

            if (env.done):
                break

            _, mark = state
            env.show_turn(True, agents, mark)

            agent = ttt_env.agent_by_mark(agents, mark)

            if (agent.indicator == "RA" or agent.indicator == "HA"):
                action = agent.act(env.available_actions())
            elif (agent.indicator == "CA"):
                action = agent.act(state, env.available_actions())
            else:
                action = agent.act(state)

            state, reward, done, info = env.step(action)
            env.render()
        
        env.show_result(True, agents, reward)
        start_mark = ttt_env.next_mark(start_mark)

    return 0


def ttt_train(max_episode=1):
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

            ttt_algos.qlearnUpdate(agents[0].qtable, prev_state, state, transition_action, score)

            prev_state = state
            transition_action = new_action

        ttt_algos.qlearnUpdate(agents[0].qtable, prev_state, None, transition_action, score)

        # Show the game's result and swap the start mark for the next episode
        start_mark = ttt_env.next_mark(start_mark)

    return agents[0].qtable


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
            playersQC.qtable = ttt_train(opt.ttt_epochs)
            metrics = ttt_play(playersQC, opt.episodes)
            return metrics
        case "QM":
            playersQM = [ttt_agents.QLearningAgent('O'), ttt_agents.MinimaxAgent('X', True)]
            playersQM.qtable = ttt_train(opt.ttt_epochs)
            metrics = ttt_play(playersQM, opt.episodes)
            return metrics
        case _:
            print("Error: No matching players")


def c4_play(env: c4_env.ConnectFourEnv, players, max_episode=1):
    start_player = 1

    for _ in range(max_episode):
        result = env.run_game(
            players[0], players[1], env.board, True, start_player)

        start_player *= -1
    
    return 0


def c4_train(env: c4_env.ConnectFourEnv, max_episode=1):
    player = c4_agents.QlearnPlayer(env, 1, 'QlearnPlayer')
    opponent = c4_agents.RandomPlayer(env, -1, 'OpponentRandomPlayer')

    players = [player, opponent]

    start_player = 1

    for _ in range(max_episode):
        result = env.train_game(
            players[0], players[1], env.board, True, start_player)
        start_player *= -1

    return players[0].qtable


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
            playersQC.qtable = c4_train(env, opt.c4_epochs)
            metrics = c4_play(env, playersQC, opt.episodes)
            return metrics
        case "QM":
            playersQM = [c4_agents.QlearnPlayer(env, 1, "QlearnPlayer"), c4_agents.MinimaxHeuristicPlayer(env, -1, True, "OpponentMinimaxHeuristicPlayer")]
            playersQM[0].qtable = c4_train(env, opt.c4_epochs)
            metrics = c4_play(env, playersQM, opt.episodes)
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



if __name__ == '__main__':
    # Parse the command line args
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='ttt', help='Which game to play?')
    parser.add_argument('--players', type=str, default='MPC', help='Which agents are playing?')
    parser.add_argument('--episodes', type=int, default=10, help='How many game rounds?')
    parser.add_argument('--ttt_epochs', type=int, default=50000, help='How long will ttt train?')
    parser.add_argument('--c4_epochs', type=int, default=10000, help='How long will c4 train?')
    args = parser.parse_args()

    if args.game == "ttt":
        ttt_flow(args)
    elif args.game == "c4":
        c4_flow(args)
    elif args.game == "all":
        all_flow(args)
    else:
        print("Error: No matching game")
