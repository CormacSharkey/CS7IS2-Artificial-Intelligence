import gym_connect_four.envs.connect_four_env as gym_env
from c4_agents import RandomPlayer, CleverPlayer, MinimaxPlayer, MinimaxPrunePlayer, MinimaxHeuristicPlayer, MinimaxPruneHeuristicPlayer, QlearnPlayer
import gym
import time

def play(env: gym_env.ConnectFourEnv, players, max_episode=1):
    # env: gym_env.ConnectFourEnv = gym.make("ConnectFour-v0")

    # player = RandomPlayer(env, 1, 'RandomPlayer')
    # opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')
    # opponent = CleverPlayer(env, -1, 'OpponentCleverPlayer')

    # player = MinimaxPlayer(env, 1, True, 'MinimaxPlayer')
    # opponent = MinimaxPlayer(env, -1, False, 'OpponentMinimaxPlayer')

    # player = MinimaxPrunePlayer(env, 1, True, 'MinimaxPrunePlayer')
    # opponent = MinimaxPrunePlayer(env, -1, False, 'OpponentMinimaxPrunePlayer')

    # player = MinimaxHeuristicPlayer(env, 1, True, 'MinimaxHeuristicPlayer')
    # opponent = MinimaxHeuristicPlayer(env, -1, False, 'OpponentMinimaxHeuristicPlayer')
    # opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')
    # opponent = CleverPlayer(env, -1, 'OpponentCleverPlayer')#

    # player = MinimaxPruneHeuristicPlayer(env, 1, True, 'MinimaxPruneHeuristicPlayer')
    # opponent = MinimaxPruneHeuristicPlayer(env, -1, False, 'OpponentMinimaxPruneHeuristicPlayer')

    # player = QlearnPlayer(env, 1, 'QlearnPlayer')
    # opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')

    # players = [player, opponent]

    start_player = 1

    for _ in range(max_episode):
        result = env.run_game(players[0], players[1], env.board, True, start_player)
        print(f"Game Over: {result}")

        start_player *= -1

def train(env: gym_env.ConnectFourEnv, max_episode=1):
    # env: gym_env.ConnectFourEnv = gym.make("ConnectFour-v0")

    player = QlearnPlayer(env, 1, 'QlearnPlayer')
    opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')

    players = [player, opponent]

    start_player = 1

    for _ in range(max_episode):
        result = env.train_game(players[0], players[1], env.board, True, start_player)
        start_player *= -1

    return players[0].qtable


if __name__ == '__main__':
    env: gym_env.ConnectFourEnv = gym.make("ConnectFour-v0")

    qtable = train(env, 50000)

    player = QlearnPlayer(env, 1, 'QlearnPlayer')
    player.qtable = qtable
    opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')

    players = [player, opponent]
    
    play(env, players, 5)