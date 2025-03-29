import gym_connect_four.envs.connect_four_env as gym_env
from c4_agents import RandomPlayer, CleverPlayer
import gym
import time

def play(max_episode=1):
    env: gym_env.ConnectFourEnv = gym.make("ConnectFour-v0")

    player = RandomPlayer(env, 1, 'RandomPlayer')
    # opponent = RandomPlayer(env, -1, 'OpponentRandomPlayer')
    opponent = CleverPlayer(env, -1, 'OpponentRandomPlayer')

    players = [player, opponent]

    start_player = 1

    for _ in range(max_episode):
        result = env.run_game(players[0], players[1], env.board, True, start_player)
        print(f"Game Over: {result}")

        start_player *= -1

if __name__ == '__main__':
    play(5)