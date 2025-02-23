import maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp
import random
import time
import numpy as np


def show_search_path(maze: gym.MazeEnv, path):
    agent = maze.maze_view.goal

    true_path = {}

    while (agent != maze.maze_view.entrance).any():
        true_path[tuple(path[tuple(agent)])] = agent
        agent = path[tuple(agent)]
        maze.maze_view._MazeView2D__colour_explored_cell(agent, (123, 3, 35), 180)
        maze.render()

    return true_path


def show_mdp_path(maze: gym.MazeEnv, path):
    agent = maze.maze_view.entrance

    directions = ["N", "E", "S", "W"]

    true_path = {}

    maze.render()
    

    # for dir in range(2, 4):
    #     if maze.maze_view.maze.is_open(agent, directions[dir]):
    #         true_path[(agent[0], agent[1])] = (agent + np.array(maze.maze_view.maze.COMPASS[directions[dir]]))
    #         agent = (agent + np.array(maze.maze_view.maze.COMPASS[directions[dir]]))
    #         maze.maze_view._MazeView2D__colour_explored_cell(agent, (123, 3, 35), 180)
    #         maze.render()
    #         break

    while (agent != maze.maze_view.goal).any():
        dir = path[agent[0]][agent[1]]
        agent = (agent + np.array(maze.maze_view.maze.COMPASS[directions[int(dir)]]))
        maze.maze_view._MazeView2D__colour_explored_cell(agent, (123, 3, 35), 180)
        maze.dash(maze.maze_view.goal)
        maze.render()

    # maze.render()

    return true_path



def main():
    # Setup the maze environment and render it
    # maze = gym.MazeEnv(maze_file="maze2d_10x10.npy", maze_size=(15, 15))
    maze = gym.MazeEnv(maze_size=(10, 10), mode="plus")
    # maze = gym.MazeEnv(maze_size=(10, 10))
    maze.render()

    # Initialized total reward
    total_reward = 0

    solved = False

    # Call the search algorithm
    solved, path = search.breadth_first_search(maze)
    show_search_path(maze, path)
    maze.reset()
    time.sleep(5)
    maze.render()
    solved, path = search.depth_first_search(maze)
    show_search_path(maze, path)
    maze.reset()
    time.sleep(5)
    maze.render()
    solved, path = search.a_star(maze)
    show_search_path(maze, path)
    maze.reset()
    time.sleep(5)
    maze.render()
    solved, agent_dirs = mdp.mdp_value_iteration(maze)
    show_mdp_path(maze, agent_dirs)
    maze.reset()
    time.sleep(5)
    maze.render()
    # solved, agent_dirs = mdp.mdp_policy_iteration(maze)
    # show_mdp_path(maze, agent_dirs)
    # maze.reset()
    # time.sleep(5)
    # maze.render()

    # If the algorithm solved the maze, print a notification and show the path
    if (solved):
        print("Well done! Maze solved!")

    # If the algorithm did not solve it (should be impossible), print a notification
    else:
        print("Oh No! Maze unsolved!")

    input("Enter any key to quit.")


if __name__ == "__main__":
    main()
