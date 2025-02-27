import time
import maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp


def benchmark_search(name, runs, algorithm, display, size, is_render):
    explored_len = []
    path_len = []
    solve_time = []
    memory_footprint = []

    for z in range(runs):
        maze = gym.MazeEnv(maze_size=(size, size), mode="plus", enable_render=is_render)
        T_start = time.monotonic()
        solved, path, explored, mem_footprint = algorithm(maze, is_render)
        T_end = time.monotonic()

        path_length = display(maze, path, is_render)
        explored_len.append(len(explored))
        path_len.append(path_length)
        solve_time.append(T_end-T_start)
        memory_footprint.append(mem_footprint)

        del maze

    print(f"Algorithm: {name} \nSolve Time Avg: {sum(solve_time)/len(solve_time): .2f} \nExplored Tiles Avg: {sum(explored_len)/len(explored_len): .2f} \nPath Tiles Avg: {sum(path_len)/len(path_len): .2f} \nMemory Footprint Avg: {sum(memory_footprint)/len(memory_footprint): .2f}\n")


def benchmark_mdp(name, runs, algorithm, display, size, is_render):
    iteration_len = []
    path_len = []
    solve_time = []
    memory_footprint = []

    for z in range(runs):
        maze = gym.MazeEnv(maze_size=(size, size), mode="plus", enable_render=is_render)
        T_start = time.monotonic()
        solved, path, iterations, mem_footprint = algorithm(maze, is_render)
        T_end = time.monotonic()

        path_length = display(maze, path, is_render)
        iteration_len.append(iterations)
        path_len.append(path_length)
        solve_time.append(T_end-T_start)
        memory_footprint.append(mem_footprint)

        del maze

    print(f"Algorithm: {name} \nSolve Time Avg: {sum(solve_time)/len(solve_time): .2f} \nIterations Avg: {sum(iteration_len)/len(iteration_len): .2f} \nPath Tiles Avg: {sum(path_len)/len(path_len): .2f}\nMemory Footprint Avg: {sum(memory_footprint)/len(memory_footprint): .2f}\n")


#! Metrics

#! Apply all metrics to mazes of varied size (10 -> 50...)
#* All Algorithms
# Time taken to solve maze
# No. of tiles used in the final solution

#* Search Algorithms
# No. of tiles explored to find a solution

#* MDP Algorithms
# Iterations before convergence
# The effects of varying gamma 




if __name__ == "__main__":
    benchmark_search("DFS", 10, search.depth_first_search, search.show_search_path, 15, False)
    benchmark_search("BFS", 10, search.depth_first_search, search.show_search_path, 15, False)
    benchmark_search("A*", 10, search.a_star, search.show_search_path, 15, False)

    benchmark_mdp("MDP Value Iteration", 20, mdp.mdp_value_iteration, mdp.show_mdp_path, 20, False)
    benchmark_mdp("MDP Policy Iteration", 20, mdp.mdp_policy_iteration, mdp.show_mdp_path, 20, False)