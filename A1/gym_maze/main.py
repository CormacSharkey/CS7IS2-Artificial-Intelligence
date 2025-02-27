import maze_src.maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp
import benchmark as bench
import time


def main(size):

    maze = gym.MazeEnv(maze_size=(size, size), mode="plus", enable_render=True)

    print("Starting Depth First Search...")
    solved, path, explored, mem_footprint = search.depth_first_search(maze, True)
    path_length = search.show_search_path(maze, path, True)
    print("Maze Solved!")
    maze.reset()
    time.sleep(5)

    print("Starting Breadth First Search...")
    solved, path, explored, mem_footprint = search.breadth_first_search(maze, True)
    path_length = search.show_search_path(maze, path, True)
    print("Maze Solved!")
    maze.reset()
    time.sleep(5)

    print("Starting A* Search...")
    solved, path, explored, mem_footprint = search.a_star(maze, True)
    path_length = search.show_search_path(maze, path, True)
    print("Maze Solved!")
    maze.reset()
    time.sleep(5)

    print("Starting MDP Value Iteration...")
    solved, path, iterations, mem_footprint = mdp.mdp_policy_iteration(maze, True)
    path_length = mdp.show_mdp_path(maze, path, True)
    print("Maze Solved!")
    maze.reset()
    time.sleep(5)

    print("Starting MDP Policy Iteration...")
    solved, path, iterations, mem_footprint = mdp.mdp_value_iteration(maze, True)
    path_length = mdp.show_mdp_path(maze, path, True)
    print("Maze Solved!")
    maze.reset()
    time.sleep(5)

    del maze

    ans = input("Press any key to quit")

if __name__ == "__main__":
    main(20)