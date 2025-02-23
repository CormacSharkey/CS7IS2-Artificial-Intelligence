from collections import deque
import maze_env as gym
import numpy as np
import time
import heapq
from itertools import count
import pygame

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

def bellman_equation(maze: gym.MazeEnv, probability, gamma, dir, values, curr_state):
    future_state = (curr_state + np.array(maze.maze_view.maze.COMPASS[dir]))

    if (maze.maze_view.maze.is_open(curr_state, dir)):
        next_state_value = values[future_state[0]][future_state[1]]
    else:
        next_state_value = 0
    
    if (future_state == maze.maze_view.goal).all():
        reward = 1
    else:
        reward = 0
        
    next_state_sum = probability*(reward + (gamma*next_state_value))

    return next_state_sum

# def get_state_value(maze: gym.MazeEnv, dir, values, curr_state):
#     future_state = (curr_state + np.array(maze.maze_view.maze.COMPASS[dir]))

#     if (maze.maze_view.maze.is_open(curr_state, dir)):
#         return values[future_state[0]][future_state[1]]
#     else:
#         return 0
    
def get_best_action(values):
    maximum = values[0]
    index = 0

    for i in range(1, 4):
        if (values[i] >= maximum):
            maximum = values[i]
            index = i

    return index, maximum


def mdp_value_iteration(maze: gym.MazeEnv):
    # Discount value (gamma)
    gamma = 0.9

    directions = ["N", "E", "S", "W"]

    # Solve state boolean
    solved = False

    values = np.zeros((maze.maze_size[0], maze.maze_size[1]))
    values[maze.maze_size[0]-1, maze.maze_size[1]-1] = 100

    agent_dirs = np.zeros((maze.maze_size[0], maze.maze_size[1]))

    iteration = 0
    theta = 0.000001

    while True:
        delta = 0
        for x in reversed(range(maze.maze_size[0])):
            for y in reversed(range(maze.maze_size[1])):
                # if not (x == 0 and y == 0) and not (x == maze.maze_size[0]-1 and y == maze.maze_size[1]-1):
                if not (x == maze.maze_size[0]-1 and y == maze.maze_size[1]-1):

                    actions = []

                    for dir in range(4):
                        forward = bellman_equation(maze, 0.8, gamma, directions[dir % 4], values, ([x, y]))
                        right = bellman_equation(maze, 0.1, gamma, directions[(dir+1) % 4], values, ([x, y]))
                        left = bellman_equation(maze, 0.1, gamma, directions[(dir-1) % 4], values, ([x, y]))

                        actions_sum = forward + right + left
                        actions.append(actions_sum)

                    value = values[x][y]

                    # Use arrays of actions to plot final path
                    index, values[x][y] = get_best_action(actions)
                    agent_dirs[x][y] = index
                    
                    # values[x][y] = max(actions)

                    delta = max(delta, abs(value-values[x][y]))

        maze.render_mdp(values=values, directions=directions, actions=agent_dirs)

        if delta < theta:
            solved = True
            break
        
        iteration += 1

    print(iteration)

    # Return the solved status
    # In theory, if there is a solution to the given maze, will always return true
    return solved, agent_dirs

#! MDP Policy Iteration


def mdp_policy_iteration(maze: gym.MazeEnv):
    # Solve state boolean
    solved = False

    # Return the solved status
    # In theory, if there is a solution to the given maze, will always return true
    return solved
