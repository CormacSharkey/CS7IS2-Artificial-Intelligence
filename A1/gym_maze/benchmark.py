import time
import maze_src.maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp
import csv
import argparse


#! Benchmark Search
# Benchmark the Search Algorithms (DFS, BFS, A*)
def benchmark_search(maze, name, algorithm, display, data, is_render):
    # Get the time before the algorithm starts
    T_start = time.monotonic()
    # Call the specified search algorithm, using the provided algorithm parameter
    solved, path, explored, mem_footprint = algorithm(maze, is_render)
    # Get the time after algorithm finishes
    T_end = time.monotonic()

    # Display the path found by the algorithm
    # This returns the length of the found path
    path_length = display(maze, path, is_render)

    # Add the solve time, explored nodes, path nodes and memory footprint to a data dictionary
    data["Average Solve Time"] += T_end-T_start
    data["Average Explored Tiles/Iterations"] += len(explored)
    data["Average Path Tiles"] += path_length
    data["Average Memory Footprint"] += mem_footprint


#! Benchmark MDP
# Benchmark the MDP Algorithms (Value Iteration, Policy Iteration)
def benchmark_mdp(maze, name, algorithm, display, data, is_render):
    # Get the time before the algorithm starts
    T_start = time.monotonic()
    # Call the specified MDP algorithm, using the provided algorithm parameter
    solved, path, iterations, mem_footprint = algorithm(
        maze, is_render=is_render)
    # Get the time after algorithm finishes
    T_end = time.monotonic()

    # Display the path found by the algorithm
    # This returns the length of the found path
    path_length = display(maze, path, is_render)

    # Add the solve time, iterations, path nodes and memory footprint to a data dictionary
    data["Average Solve Time"] += T_end-T_start
    data["Average Explored Tiles/Iterations"] += iterations
    data["Average Path Tiles"] += path_length
    data["Average Memory Footprint"] += mem_footprint

#! Benchmark
# Benchmark the Search and MDP Algorithms
# Runs the specified number of times with a maze of a specified dimension
# Also saves the results to a CSV file


def benchmark(runs, size):
    # Specify the field names for the CSV file
    field_names = ['Average Solve Time', 'Average Explored Tiles/Iterations',
                   'Average Path Tiles', 'Average Memory Footprint']

    # Set up dictionaries to store the data for each algorithm, setting the key values to 0
    dfs_data = {key: 0 for key in field_names}
    bfs_data = {key: 0 for key in field_names}
    a_star_data = {key: 0 for key in field_names}
    mdp_vi_data = {key: 0 for key in field_names}
    mdp_pi_data = {key: 0 for key in field_names}

    # Loop through the specified number of runs
    for z in range(runs):
        # Create a new maze of specified size
        maze = gym.MazeEnv(maze_size=(size, size),
                           mode="plus", enable_render=False)
        # Benchmark DFS
        benchmark_search(maze, "DFS", search.depth_first_search,
                         search.show_search_path, dfs_data, False)
        # Reset the maze
        maze.reset()
        # Benchmark BFS
        benchmark_search(maze, "BFS", search.breadth_first_search,
                         search.show_search_path, bfs_data, False)
        # Reset the maze
        maze.reset()
        # Benchmark A*
        benchmark_search(maze, "A*", search.a_star,
                         search.show_search_path, a_star_data, False)
        # Reset the maze
        maze.reset()
        # Benchmark MDP Value Iteration
        benchmark_mdp(maze, "MDP Value Iteration", mdp.mdp_value_iteration,
                      mdp.show_mdp_path, mdp_vi_data, False)
        # Reset the maze
        maze.reset()
        # Benchmark MDP Policy Iteration
        benchmark_mdp(maze, "MDP Policy Iteration",
                      mdp.mdp_policy_iteration, mdp.show_mdp_path, mdp_pi_data, False)

        # Delete the maze
        del maze

    # Set up a list to store the data for each algorithm
    data = []
    # Set up a list to store each row of data
    row = []

    # For each key in the DFS data dictionary
    for key in dfs_data:
        # Append the average of the value to the data list
        data.append(dfs_data[key] / runs)
        # Append the data list to the row list
    row.append(data)

    # Print the metrics for the DFS algorithm
    print(
        f"Algorithm: DFS \nSolve Time Avg: {data[0]: .3f} \nExplored Tiles Avg: {data[1]: .3f} \nPath Tiles Avg: {data[2]: .3f} \nMemory Footprint Avg: {data[3]: .3f}\n")

    # Reset the data list
    data = []

    # For each key in the BFS data dictionary
    for key in bfs_data:
        # Append the average of the value to the data list
        data.append(bfs_data[key] / runs)
    # Append the data list to the row list
    row.append(data)

    # Print the metrics for the BFS algorithm
    print(
        f"Algorithm: BFS \nSolve Time Avg: {data[0]: .3f} \nExplored Tiles Avg: {data[1]: .3f} \nPath Tiles Avg: {data[2]: .3f} \nMemory Footprint Avg: {data[3]: .3f}\n")

    # Reset the data lisT
    data = []

    # For each key in the A* data dictionary
    for key in a_star_data:
        # Append the average of the value to the data list
        data.append(a_star_data[key] / runs)
    # Append the data list to the row list
    row.append(data)

    # Print the metrics for the A* algorithm
    print(
        f"Algorithm: A* \nSolve Time Avg: {data[0]: .3f} \nExplored Tiles Avg: {data[1]: .3f} \nPath Tiles Avg: {data[2]: .3f} \nMemory Footprint Avg: {data[3]: .3f}\n")

    # Reset the data list
    data = []

    # For each key in the MDP Value Iteration data dictionary
    for key in mdp_vi_data:
        # Append the average of the value to the data list
        data.append(mdp_vi_data[key] / runs)
    # Append the data list to the row list
    row.append(data)

    # Print the metrics for the MDP Value Iteration algorithm
    print(
        f"Algorithm: MDP Value Iteration \nSolve Time Avg: {data[0]: .3f} \nIterations Avg: {data[1]: .3f} \nPath Tiles Avg: {data[2]: .3f} \nMemory Footprint Avg: {data[3]: .3f}\n")

    # Reset the data list
    data = []

    # For each key in the MDP Policy Iteration data dictionary
    for key in mdp_pi_data:
        # Append the average of the value to the data list
        data.append(mdp_pi_data[key] / runs)
    # Append the data list to the row list
    row.append(data)

    # Print the metrics for the MDP Policy Iteration algorithm
    print(
        f"Algorithm: MDP Policy Iteration \nSolve Time Avg: {data[0]: .3f} \nIterations Avg: {data[1]: .3f} \nPath Tiles Avg: {data[2]: .3f} \nMemory Footprint Avg: {data[3]: .3f}\n")

    # Write the data to a CSV file
    with open(f"A1/data/data{size}.csv", 'w') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write the field names to the CSV file
        csvwriter.writerow(field_names)

        # Write the row list to the CSV file
        csvwriter.writerows(row)


if __name__ == "__main__":
    # Setup an argument parser and specify two arguments for runs and size
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10, help='The number of benchmark runs')
    parser.add_argument('--size', type=int, default=15, help='The maze size')

    # Parse the arguments
    opt = parser.parse_args()

    # Print a starting message to inform the user of the benchmark
    print(f"Benchmarking maze size {opt.size} for {opt.runs} runs...")

    # Run the benchmark function with the specified number of runs and maze
    benchmark(opt.runs, opt.size)
