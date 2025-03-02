import argparse
import maze_src.maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp
import time


def main(opt):
    # Create a new maze with the specified size
    maze = gym.MazeEnv(maze_size=(opt.size, opt.size), mode="plus", enable_render=True)

    # Match-Case
    # If the provided argument matches any algorithm keyname, run that algorithm
    # Default case is to run all algorithms with 5 second gaps in between
    match opt.algo:
        # DFS case
        case "dfs":
            print("Starting Depth First Search...")
            solved, path, explored, mem_footprint = search.depth_first_search(maze, is_render=True)
            path_length = search.show_search_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()

        # BFS case
        case "bfs":
            print("Starting Breadth First Search...")
            solved, path, explored, mem_footprint = search.breadth_first_search(maze, is_render=True)
            path_length = search.show_search_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()

        # A* case
        case "a*":
            print("Starting A* Search...")
            solved, path, explored, mem_footprint = search.a_star(maze, is_render=True)
            path_length = search.show_search_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()
        
        # MDP Value Iteration case
        case "vi":
            print("Starting MDP Value Iteration...")
            solved, path, iterations, mem_footprint = mdp.mdp_value_iteration(maze, is_render=True)
            path_length = mdp.show_mdp_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()

        # MDP Policy Iteration case
        case"pi": 
            print("Starting MDP Policy Iteration...")
            solved, path, iterations, mem_footprint = mdp.mdp_policy_iteration(maze, is_render=True)
            path_length = mdp.show_mdp_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()

        # Default case
        case _:
            print("Starting Depth First Search...")
            solved, path, explored, mem_footprint = search.depth_first_search(maze, is_render=True)
            path_length = search.show_search_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()
            time.sleep(5)

            print("Starting Breadth First Search...")
            solved, path, explored, mem_footprint = search.breadth_first_search(maze, is_render=True)
            path_length = search.show_search_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()
            time.sleep(5)

            print("Starting A* Search...")
            solved, path, explored, mem_footprint = search.a_star(maze, is_render=True)
            path_length = search.show_search_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()
            time.sleep(5)

            print("Starting MDP Value Iteration...")
            solved, path, iterations, mem_footprint = mdp.mdp_value_iteration(maze, is_render=True)
            path_length = mdp.show_mdp_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()
            time.sleep(5)

            print("Starting MDP Policy Iteration...")
            solved, path, iterations, mem_footprint = mdp.mdp_policy_iteration(maze, is_render=True)
            path_length = mdp.show_mdp_path(maze, path, True)
            print(f"Maze solved in {path_length} steps!")
            maze.reset()
            time.sleep(5)

    ans = input("Press any key to quit")
    del maze

if __name__ == "__main__":
    # Setup an argument parser and specify two arguments for algorithm keyname and size
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="", help='The maze solving algorithm to use')
    parser.add_argument('--size', type=int, default=15, help='The maze size')

    # Parse the arguments
    opt = parser.parse_args()
    
    # Run main function with the parsed arguments
    main(opt)