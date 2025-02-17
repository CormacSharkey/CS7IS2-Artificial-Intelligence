import maze_env as mze
import algorithms.search as search
import random


def main():
    # Setup the maze environment and render it
    maze = mze.MazeEnv(maze_size=(15, 15))
    maze.render()

    # Initialized total reward
    total_reward = 0

    # Call the search algorithm
    # solved = search.breadth_first_search(maze)
    solved = search.depth_first_search(maze)

    # If the algorithm solved the maze, print a notification and show the path
    if (solved):
        print("Well done! Maze solved!")

        # for node in path:
        #     maze.maze_view._MazeView2D__colour_cell(node, (10, 128, 64), 200)
        #     maze.render()
    
    # If the algorithm did not solve it (should be impossible), print a notification
    else:
        print("Oh No! Maze unsolved!")

    input("Enter any key to quit.")


if __name__ == "__main__":
    main()
