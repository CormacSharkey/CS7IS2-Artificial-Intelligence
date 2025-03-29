# import gym
import gym_connect4.envs.connect4_env as connect4
import random
from connect4_agents import CleverAgent, RandomAgent, MinimaxRawAgent, MinimaxHeuristicAgent


def play(max_episode=1):

    # env = gym.make('Connect4Env-v0')
    env = connect4.Connect4Env()

    # agents = [RandomAgent(0), CleverAgent(1)]
    agents = [MinimaxHeuristicAgent(0, True), MinimaxHeuristicAgent(1, False)]
    # agents = [MinimaxHeuristicAgent(0, True), CleverAgent(1)]

    for _ in range(max_episode):
        obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
        game_over = False

        while not game_over:
            action_dict = {}

            # # If it's player 1's turn {0, 1}
            # if env.game.player == 1:
            #     action_dict[0] = agents[0].act(env)
            #     action_dict[1] = 0

            # # Else it's player 2's turn
            # else:
            #     action_dict[0] = 0
            #     action_dict[1] = agents[1].act(env)

            board = env.game.clone()

            action_dict[env.game.player] = 0
            action_dict[env.game.player ^ 1] = agents[env.game.player ^ 1].act(board)

            obses, rewards, game_over, info = env.step(action_dict)
            env.render()
            print("Stepped!")
        
        if (env._get_winner() == -1):
            print("It's a draw!")
        elif (env._get_winner() == 0):
            print("Player 1 wins!")
        elif (env._get_winner() == 1):
            print("Player 2 wins!")

        # switch the player order (not who puts down 1 and 2 on the board)

if __name__ == '__main__':
    play(1)
