import time
import maze_src.maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp
import csv

def benchmark_search(maze, name, algorithm, display, data, is_render):
    T_start = time.monotonic()
    solved, path, explored, mem_footprint = algorithm(maze, is_render)
    T_end = time.monotonic()

    path_length = display(maze, path, is_render)

    data["Average Solve Time"] += T_end-T_start
    data["Average Explored Tiles/Iterations"] += len(explored)
    data["Average Path Tiles"] += path_length
    data["Average Memory Footprint"] += mem_footprint

    # print(f"Algorithm: {name} \nSolve Time Avg: {sum(solve_time)/len(solve_time): .2f} \nExplored Tiles Avg: {sum(explored_len)/len(explored_len): .2f} \nPath Tiles Avg: {sum(path_len)/len(path_len): .2f} \nMemory Footprint Avg: {sum(memory_footprint)/len(memory_footprint): .2f}\n")


def benchmark_mdp(maze, name, algorithm, display, data, is_render):
    T_start = time.monotonic()
    solved, path, iterations, mem_footprint = algorithm(maze, is_render)
    T_end = time.monotonic()

    path_length = display(maze, path, is_render)

    data["Average Solve Time"] += T_end-T_start
    data["Average Explored Tiles/Iterations"] += iterations
    data["Average Path Tiles"] += path_length
    data["Average Memory Footprint"] += mem_footprint

    # print(f"Algorithm: {name} \nSolve Time Avg: {sum(solve_time)/len(solve_time): .2f} \nIterations Avg: {sum(iteration_len)/len(iteration_len): .2f} \nPath Tiles Avg: {sum(path_len)/len(path_len): .2f}\nMemory Footprint Avg: {sum(memory_footprint)/len(memory_footprint): .2f}\n")


def benchmark(runs, size):
    field_names = ['Average Solve Time', 'Average Explored Tiles/Iterations', 'Average Path Tiles', 'Average Memory Footprint']

    dfs_data = {key: 0 for key in field_names}
    bfs_data = {key: 0 for key in field_names}
    a_star_data = {key: 0 for key in field_names}
    mdp_vi_data = {key: 0 for key in field_names}
    mdp_pi_data = {key: 0 for key in field_names}


    for z in range(runs):
        maze = gym.MazeEnv(maze_size=(size, size), mode="plus")
        benchmark_search(maze, "DFS", search.depth_first_search, search.show_search_path, dfs_data, False)
        maze.reset()
        benchmark_search(maze, "BFS", search.breadth_first_search, search.show_search_path, bfs_data, False)
        maze.reset()
        benchmark_search(maze, "A*", search.a_star, search.show_search_path, a_star_data, False)
        maze.reset()
        benchmark_mdp(maze, "MDP Value Iteration", mdp.mdp_value_iteration, mdp.show_mdp_path, mdp_vi_data, False)
        maze.reset()
        benchmark_mdp(maze, "MDP Policy Iteration", mdp.mdp_policy_iteration, mdp.show_mdp_path, mdp_pi_data, False)

        del maze

    data = []
    row = []

    for key in dfs_data:
        # dfs_data[key] /= runs
        data.append(dfs_data[key] / runs)
    row.append(data)

    print(f"Algorithm: DFS \nSolve Time Avg: {data[0]: .2f} \nExplored Tiles Avg: {data[1]: .2f} \nPath Tiles Avg: {data[2]: .2f} \nMemory Footprint Avg: {data[3]: .2f}\n")

    data = []
    for key in bfs_data:
        # bfs_data[key] /= runs
        data.append(bfs_data[key] / runs)
    row.append(data)

    print(f"Algorithm: BFS \nSolve Time Avg: {data[0]: .2f} \nExplored Tiles Avg: {data[1]: .2f} \nPath Tiles Avg: {data[2]: .2f} \nMemory Footprint Avg: {data[3]: .2f}\n")

    data = []
    for key in a_star_data:
        # a_star_data[key] /= runs
        data.append(a_star_data[key] / runs)
    row.append(data)

    print(f"Algorithm: A* \nSolve Time Avg: {data[0]: .2f} \nExplored Tiles Avg: {data[1]: .2f} \nPath Tiles Avg: {data[2]: .2f} \nMemory Footprint Avg: {data[3]: .2f}\n")

    data = []
    for key in mdp_vi_data:
        # mdp_vi_data[key] /= runs
        data.append(mdp_vi_data[key] / runs)
    row.append(data)

    print(f"Algorithm: MDP Value Iteration \nSolve Time Avg: {data[0]: .2f} \nIterations Avg: {data[1]: .2f} \nPath Tiles Avg: {data[2]: .2f} \nMemory Footprint Avg: {data[3]: .2f}\n")
    
    data = []
    for key in mdp_pi_data:
        # mdp_pi_data[key] /= runs
        data.append(mdp_pi_data[key] / runs)
    row.append(data)

    print(f"Algorithm: MDP Policy Iteration \nSolve Time Avg: {data[0]: .2f} \nIterations Avg: {data[1]: .2f} \nPath Tiles Avg: {data[2]: .2f} \nMemory Footprint Avg: {data[3]: .2f}\n")
    

    # writing to csv file
    with open(f"A1/data/data{size}.csv", 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(field_names)

        # writing the data rows
        csvwriter.writerows(row)

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
    benchmark(10, 50)