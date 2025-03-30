import gym_connect_four.envs.connect_four_env as gym_env
from gym_connect_four.c4_agents import RandomPlayer, CleverPlayer, MinimaxPlayer, MinimaxPrunePlayer, MinimaxHeuristicPlayer, MinimaxPruneHeuristicPlayer, QlearnPlayer
import gym


def play(env: gym_env.ConnectFourEnv, players, max_episode=1):
    start_player = 1

    for _ in range(max_episode):
        result = env.run_game(
            players[0], players[1], env.board, True, start_player)
        print(f"Game Over: {result}")

        start_player *= -1


def train(env: gym_env.ConnectFourEnv, max_episode=1):
    player = QlearnPlayer(env, 1, 'QlearnPlayer')
    opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')

    players = [player, opponent]

    start_player = 1

    for _ in range(max_episode):
        result = env.train_game(
            players[0], players[1], env.board, True, start_player)
        start_player *= -1

    return players[0].qtable


if __name__ == '__main__':
    env: gym_env.ConnectFourEnv = gym.make("ConnectFour-v0")

    players = [QlearnPlayer(env, 1, 'QlearnPlayer'), RandomPlayer(env, -1, 'OpponentRandomPlayer')]
    
    qtable = train(env, 10000)

    print("Training Complete!")

    players[0].qtable = qtable
    play(env, players, 5)
