import gym
import gym_connect4.envs.connect4_env as connect4
import random


def random_act(env: connect4.Connect4Env, player, ava_actions):
    # Return a random action
    return random.choice(ava_actions)

def clever_act(env: connect4.Connect4Env, player, ava_actions):
    # For every action in the provided available actions
    for action in ava_actions:

        clone_env = env.game.clone()
        clone_env.player = player

        clone_env.move(action)
        if (clone_env.is_winner(clone_env.player)):
            del clone_env
            return action
        
        clone_env.player ^= 1

    for action in ava_actions:
        clone_env.move(action)
        if (clone_env.is_winner(clone_env.player)):
            del clone_env
            return action
        
    return random.choice(ava_actions)

def play():

    # env = gym.make('Connect4Env-v0')
    env = connect4.Connect4Env()

    agents = ['Agent1()', 'Agent2()']
    obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
    game_over = False
    while not game_over:
        action_dict = {}
        for agent_id, agent in enumerate(agents):

            if (agent_id == 0):
                action_dict[agent_id] = clever_act(env, agent_id, env.game.get_moves())

            else:
                action_dict[agent_id] = random_act(env, agent_id, env.game.get_moves())

        
        obses, rewards, game_over, info = env.step(action_dict)
        env.render()
    
    if (env._get_winner() == -1):
        print("It's a draw!")
    elif (env._get_winner() == 0):
        print("Player 1 wins!")
    elif (env._get_winner() == 1):
        print("Player 2 wins!")

if __name__ == '__main__':
    play()
