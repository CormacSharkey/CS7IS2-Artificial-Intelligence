import maze_src.maze_env as gym
import numpy as np
import sys


#! Bellman Equation
# Calculate the value of a given node for a given primary direction
# Apply the Bellman Equation to determine a state value for a node in a given direction
# Takes the probability of going in a direction, a discount (gamma), the direction's node's value, and the reward for the direction
def bellman_equation(maze: gym.MazeEnv, probability, gamma, dir, values, curr_state):
    # Get the neighbour state for a given direction
    future_state = np.copy(
        curr_state + np.array(maze.maze_view.maze.COMPASS[dir]))

    # If the neighbour is not blocked by a wall or out of bounds
    if (maze.maze_view.maze.is_open(curr_state, dir)):
        # Set the next state value as the neighbour's state value
        next_state_value = values[future_state[0]][future_state[1]]
    else:
        # Else, the next state value is 0 (no viable state to move to)
        next_state_value = 0

    # If the neighbour state is the goal
    if (future_state == maze.maze_view.goal).all():
        # The reward is 1
        reward = 1
    # Else, the reward is 0 (for every neighbour state not the goal)
    else:
        reward = 0

    # The next state value - Bellman Equation for one action probability
    next_state = probability*(reward + (gamma*next_state_value))

    # Return the next state value
    return next_state


#! Get Best Action
# For a given set of state values, find the max and return the corresponding action
def get_best_action(values):
    # Initialize the max value as the first value
    maximum = values[0]
    # Initialize the max value index as the first value index
    # This also indicates the action based on an assumed action array
    index = 0

    # For the remaining values
    for i in range(1, 4):
        # If the current value is greater or equal to the current max value
        if (values[i] >= maximum):
            # Update the max value to the current value
            maximum = values[i]
            # Update the max value index to the current index
            index = i

    # Return the max value index and the max value
    return index, maximum


#! Show MDP Path
# Display the true path for a given MDP algorithm on the maze in dark red
# Propagates from start to end
def show_mdp_path(maze: gym.MazeEnv, path, is_render=True):
    # Initialize the agent as being at the start
    agent = np.copy(maze.maze_view.entrance)

    # Initialize an assumed action array (from MDP algorithms)
    directions = ["N", "E", "S", "W"]
    # Initialize the path length as 0
    path_length = 0

    # Render the maze normally to keep it clean
    if (is_render):
        maze.render()

    # While the agent has not reached the goal
    while (agent != maze.maze_view.goal).any():
        # Get the policy for the agent's state
        dir = path[agent[0]][agent[1]]
        # Move the agent to the policy's indicating state
        agent = np.copy(
            agent + np.array(maze.maze_view.maze.COMPASS[directions[int(dir)]]))
        # Colour the node to indicate the true path
        maze.maze_view._MazeView2D__colour_explored_cell(
            agent, (123, 3, 35), 180)
        # Increment the path length
        path_length += 1

        # Move the agent to the next node visually
        maze.dash(agent)
        # Render the maze normally to show the coloured true path
        if (is_render):
            maze.render()

    # Return the true path length
    return path_length-1


#! MDP Value Iteration
# Set all state (node) values to 0 at iteration 0, with goal state as 100 (or 1, 10, etc...)
# Use the Bellman Equation to calculate the surrounding nodes' state values
# Repeat iterations until state values have propagated to all states and have converged
# Upon convergence, path has been calculated
def mdp_value_iteration(maze: gym.MazeEnv, gamma=0.9, is_render=True):
    # Assumed array of actions (assumption used by methods above)
    directions = ["N", "E", "S", "W"]

    # Solve state boolean
    solved = False

    # Initialize an all-zero array for the state values - make it the size of the maze
    values = np.zeros((maze.maze_size[0], maze.maze_size[1]))
    # Set the goal state value
    values[maze.maze_size[0]-1, maze.maze_size[1]-1] = 100

    # Initialize an all-zero array for the primary direction for each state
    # (index of above array represents direction)
    agent_dirs = np.full((maze.maze_size[0], maze.maze_size[1]), 1)

    # Initialize an iteration counter for metrics
    iteration = 0
    # Initialize a value of theta - a threshold for determining convergence
    theta = 0.0000001

    # While True - broken by convergence
    while True:
        # Set the current max delta between new and old state values as 0
        delta = 0

        # Copy the current state values to a temporary array for calculations
        temp_values = np.copy(values)

        # For every node (state) in the maze
        for x in (range(maze.maze_size[0])):
            for y in (range(maze.maze_size[1])):
                # If it's not the goal state
                if not (x == maze.maze_size[0]-1 and y == maze.maze_size[1]-1):

                    # Initialize an array for state values
                    actions = []

                    # For every possible direction (4)
                    for dir in range(4):
                        # Calculate the deterministic bellman sum for the given direction
                        bellman_sum = bellman_equation(
                            maze, 1, gamma, directions[dir], temp_values, ([x, y]))

                        # Add it to the direction array
                        actions.append(bellman_sum)

                    # Store the previous state value temporarily
                    value = temp_values[x][y]

                    # Use the actions array to plot the final path and update the values
                    agent_dirs[x][y], values[x][y] = get_best_action(actions)

                    # Calculate the delta to measure convergence
                    delta = max(delta, abs(value-values[x][y]))

        # Render the new state values
        if (is_render):
            maze.render_mdp(values=values, flag=False)

        # Update the iterations by 1
        iteration += 1

        # If the max delta for the last iteration is less that theta (threshold)
        # convergence has been reached
        # Break the loop
        if delta < theta:
            solved = True
            break

    # Calculate the memory footprint of the algorithm
    memory_footprint = sys.getsizeof(directions)
    + sys.getsizeof(values)
    + sys.getsizeof(temp_values)
    + sys.getsizeof(agent_dirs)
    + sys.getsizeof(actions)

    # Return the solved status, state policies, iterations and memory footprint
    # In theory, if there is a solution to the given maze, will always return true
    return solved, agent_dirs, iteration, memory_footprint

#! MDP Policy Iteration
# Policy = direction
# Start with a random policy for all states
# Calculate policy score for each state until convergence (Policy Evaluation)
# Update all states to best policy (Policy Improvement)
# Repeat until policy converges
def mdp_policy_iteration(maze: gym.MazeEnv, gamma=0.9, is_render=True):
    # Assumed array of actions (assumption used by methods above)
    directions = ["N", "E", "S", "W"]

    # Solve state boolean
    solved = False

    # Initialize an all-zero array for the state values - make it the size of the maze
    values = np.zeros((maze.maze_size[0], maze.maze_size[1]))
    # Set the goal state value
    values[maze.maze_size[0]-1, maze.maze_size[1]-1] = 100

    # Initialize an all-one array for the primary direction for each state
    # (index of above array represents direction)
    agent_dirs = np.full((maze.maze_size[0], maze.maze_size[1]), 1)

    # Initialize an iteration counter for metrics
    iteration = 0
    # Initialize a value of theta - a threshold for determining convergence
    theta = 0.0000001

    # Flag to control termination at convergence
    not_converged = True

    # While the state policies have not converged
    while not_converged:
        # While True - broken when state values converge
        while True:
            # Set the current max delta between new and old state values as 0
            delta = 0

            # Copy the current state values to a temporary array for calculations
            temp_values = np.copy(values)

            # For every node (state) in the maze
            for x in (range(maze.maze_size[0])):
                for y in (range(maze.maze_size[1])):
                    # If it's not the goal state
                    if not (x == maze.maze_size[0]-1 and y == maze.maze_size[1]-1):
                        # Calculate the deterministic bellman sum for the given direction
                        bellman_sum = bellman_equation(
                            maze, 1, gamma,
                            directions[int(agent_dirs[x][y])],
                            temp_values, ([x, y])
                        )

                        # Update the previous state value with the new state value
                        values[x][y] = bellman_sum

                        # Calculate the delta to measure convergence
                        delta = max(delta, abs(temp_values[x][y]-values[x][y]))

            # Render the new state values
            if (is_render):
                maze.render_mdp(values=agent_dirs, flag=True)

            # If the max delta for the last iteration is less that theta (threshold)
            # convergence has been reached
            # Break the loop
            if delta < theta:
                solved = True
                break

        # Assume the policies have converged
        converged_policy = True
        # For every node (state) in the maze
        for x in (range(maze.maze_size[0])):
            for y in (range(maze.maze_size[1])):
                # If it's not the goal state
                if not (x == maze.maze_size[0]-1 and y == maze.maze_size[1]-1):

                    # Initialize an array for policies
                    actions = []

                    # For every possible direction (4)
                    for dir in range(4):
                        # Calculate the deterministic bellman sum for the given direction
                        bellman_sum = bellman_equation(
                            maze, 1, gamma, directions[dir], values, ([x, y]))

                        # Add it to the policy array
                        actions.append(bellman_sum)

                    # Use arrays of policies to find the best policy for the given state
                    index, new_value = get_best_action(actions)

                    # If the previous policy has changed
                    if (index != agent_dirs[x][y]):
                        # Record that the policies have not converged
                        converged_policy = False

                    # Assign the new policy to its state
                    agent_dirs[x][y] = index

        # Render the new state values
        if (is_render):
            maze.render_mdp(values=agent_dirs, flag=True)

        # Update the iterations by 1
        iteration += 1

        # If the policies have converged, break the loop
        if (converged_policy == True):
            not_converged = False
            solved = True

    # Calculate the memory footprint of the algorithm
    memory_footprint = sys.getsizeof(directions)
    + sys.getsizeof(values)
    + sys.getsizeof(temp_values)
    + sys.getsizeof(agent_dirs)
    + sys.getsizeof(actions)

    # Return the solved status, state policies, iterations and memory footprint
    # In theory, if there is a solution to the given maze, will always return true
    return solved, agent_dirs, iteration, memory_footprint
