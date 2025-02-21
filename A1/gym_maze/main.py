import maze_env as gym
import algorithms.search as search
import random
import time


def main():
    # Setup the maze environment and render it
    # maze = gym.MazeEnv(maze_file="maze2d_10x10.npy", maze_size=(15, 15))
    # maze = gym.MazeEnv(maze_size=(20, 20), mode="plus")
    maze = gym.MazeEnv(maze_size=(20, 20))
    maze.render()

    # Initialized total reward
    total_reward = 0

    # Call the search algorithm
    solved = search.breadth_first_search(maze)
    maze.reset()
    time.sleep(5)
    solved = search.depth_first_search(maze)
    maze.reset()
    time.sleep(5)
    solved = search.a_star(maze)
    maze.reset()
    time.sleep(5)

    # If the algorithm solved the maze, print a notification and show the path
    if (solved):
        print("Well done! Maze solved!")

    # If the algorithm did not solve it (should be impossible), print a notification
    else:
        print("Oh No! Maze unsolved!")

    input("Enter any key to quit.")


if __name__ == "__main__":
    main()
