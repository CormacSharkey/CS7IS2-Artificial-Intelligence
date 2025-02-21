from collections import deque
import maze_env as gym
import numpy as np
import time
import heapq
from itertools import count

# * MDP Understanding:
#! MDP Value Iteration
# Set all state (node) values to 0 at iteration 0, with goal state as 100 (or 1, 10, etc...)
# Use the Bellman Equation to calculate the surrounding nodes' state values
# Repeat iterations until state values have propagated to all states and have converged
# Upon convergence, path has been calculated

#! MDP Policy Iteration
# Policy = direction
# Start with a random policy for all states
# Calculate policy score for each state (Policy Evaluation)
# Perform value iteration until convergence (Policy Improvement)
# Repeat until policy converges

#! MDP Value Iteration


def mdp_value_iteration(maze: gym.MazeEnv):
    # Solve state boolean
    solved = False

    # Return the solved status
    # In theory, if there is a solution to the given maze, will always return true
    return solved

#! MDP Policy Iteration


def mdp_policy_iteration(maze: gym.MazeEnv):
    # Solve state boolean
    solved = False

    # Return the solved status
    # In theory, if there is a solution to the given maze, will always return true
    return solved
