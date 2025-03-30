from gym.envs.registration import register
from .envs.connect_four_env import ConnectFourEnv, ResultType

register(
    id='ConnectFour-v0',
    entry_point='c4.gym_connect_four.envs.connect_four_env:ConnectFourEnv',
)