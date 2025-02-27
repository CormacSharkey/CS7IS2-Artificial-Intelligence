from collections import deque
import maze_src.maze_env as gym
import numpy as np
import sys
import heapq
from itertools import count


#! Check Neighbours
# For the current agent state (node), check if any surrounding neighbours (N, E, S, W) are valid
# If they are, add them to a queue of next steps and return the queue
# Valid = the neighbour has never been visited before and there is no wall between the agent's node and the neighbour
def check_neighbour(maze: gym.MazeEnv, visited: deque, search: deque, path):
    # Create a queue of the next steps
    next_steps = deque()

    # For every direction available (N, E, S, W)
    for dir in maze.ACTION:
        # Calculate the neighbour given a direction
        future_state = np.copy(maze.maze_view.robot +
                        np.array(maze.maze_view.maze.COMPASS[dir]))
        # If the neighbour has not been visited and there is no wall between the agent's node and the neighbour
        if (not any((future_state == elem).all() for elem in visited)) and (not any((future_state == elem).all() for elem in search))and (maze.maze_view.maze.is_open(maze.maze_view.robot, dir)):
            # Add it to the next steps queue
            next_steps.append(future_state)

            # Link the future node to the current node for plotting the final path
            path[tuple(future_state)] = np.copy(maze.maze_view.robot)

    # After checking all possible neighbours, return the queue
    return next_steps


#! Calculate F
# Calculate the f score for a given neighbour
# F = G + H
# G = cost of the current path up to this point from the start
# H = heuristic distance to the goal
def calculate_f(maze: gym.MazeEnv, running_g, future_node):
    # The value of g is given as a param
    new_g = running_g
    # The value of h is calculated as the Manhattan Distance from the neighbour to the goal (distance x + distance y)
    # Manhattan Distance
    new_h = np.sum(np.abs(maze.maze_view.goal - future_node))
    # Euclidean Distance
    # new_h = int(np.linalg.norm(maze.maze_view.goal - future_node)) 

    # The value of f is the sum of g and h
    new_f = new_g + new_h

    # Return the calculated value of f
    return new_f


#! Move Agent
# Move the agent to a given node and update the maze display
# Given a destination node and colour, move the agent to the destination and colour it to show it's path of exploration
# Also, re-render the maze to update the display for the viewer
def move_agent(maze: gym.MazeEnv, node, colour, is_render=True):
    # Move the agent to the given node
    state, reward, done, info = maze.dash(node)
    # If-Else statement - colour the agent's exploration path differently depending on which search algorithm is active
    if (colour == "purple"):
        maze.maze_view._MazeView2D__colour_explored_cell(
            state, (180, 0, 180), 180)
    elif (colour == "gold"):
        maze.maze_view._MazeView2D__colour_explored_cell(
            state, (187, 165, 61), 180)
    elif (colour == "cyan"):
        maze.maze_view._MazeView2D__colour_explored_cell(
            state, (0, 139, 139), 180)
    # Render the maze - update the display with the agent's movement and path colouring for the viewer
    if (is_render):
        maze.render()

    # Return the agent's current node, the current reward, and whether the agent has reached the goal or not
    return state, reward, done


#! Show Search Path
# Display the true path for a given search algorithm on the maze in dark red
def show_search_path(maze: gym.MazeEnv, path, is_render=True):
    # Initialize the agent as being at the goal
    agent = np.copy(maze.maze_view.goal)

    # Initialize the path length as 0
    path_length = 0

    # While the agent has not reached the start
    while (agent != maze.maze_view.entrance).any():
        # Set the agent as being at the next node on the true path
        agent = np.copy(path[tuple(agent)])
        # Colour the node to indicate the true path
        maze.maze_view._MazeView2D__colour_explored_cell(
            agent, (123, 3, 35), 180)
        # Increment the path length
        path_length += 1
        
        # Render the true path as it is coloured
        if (is_render):
            maze.render()

    # Return the true path length
    return path_length-1


#! DFS Algorithm - LIFO
# Start at beginning point
# Check if any neighbours are valid to move to
# Add valid neighbours to stack
# Pop entry off stack and move to it
# Repeat until goal reached
def depth_first_search(maze: gym.MazeEnv, is_render=True):
    # Solve state boolean
    solved = False

    # Visited queue - nodes the agent has already visited
    visited_nodes = deque()
    # Add the start node to the visited queue
    visited_nodes.append(maze.maze_view.entrance)

    # Path dictionary - determined path stored through node linking
    path = {}

    # Search stack - nodes the agent is planning on visiting
    search_moves = deque()
    # Get a queue of valid neighbours the agent can visit from the current node
    next_steps = check_neighbour(maze, visited_nodes, search_moves, path)

    # Add the neighbours to the search stack
    search_moves.extend(next_steps)

    # While there are nodes in the search queue - there are places to visited
    while (len(search_moves) != 0):
        # Pop the next node from the search stack - stack pop
        next_node = search_moves.pop()

        # Move the agent to the next node and colour the path
        state, reward, done = move_agent(maze, next_node, "purple", is_render)

        visited_nodes.append(maze.maze_view.robot)

        # Get a queue of valid neighbours the agent can visit from the current node
        next_steps = check_neighbour(maze, visited_nodes, search_moves, path)

        # If there are valid neighbours
        if (len(next_steps) != 0):
            # Add them to the search stack
            search_moves.extend(next_steps)

        # If the current node is the goal, then the maze is solved
        # Set solved status as True
        # Break out of the search loop
        if (done):
            solved = True
            break

    memory_footprint = sys.getsizeof(visited_nodes) + sys.getsizeof(search_moves) + sys.getsizeof(path)

    # Return the solved status and the final path
    # In theory, if there is a solution to the given maze, will always return true
    return solved, path, visited_nodes, memory_footprint


#! BFS Algorithm - FIFO
# Start at beginning point
# Check if any neighbours are valid to move to
# Add valid neighbours to queue
# Pop entry off queue and move to it
# Repeat until goal reached
def breadth_first_search(maze: gym.MazeEnv, is_render=True):
    # Solve state boolean
    solved = False

    # Visited queue - nodes the agent has already visited
    visited_nodes = deque()
    # Add the start node to the visited queue
    visited_nodes.append(maze.maze_view.entrance)

    # Path dictionary - determined path stored through node linking
    path = {}

    # Search queue - nodes the agent is planning on visiting
    search_moves = deque()
    # Get a queue of valid neighbours the agent can visit from the current node
    next_steps = check_neighbour(maze, visited_nodes, search_moves, path)

    # Add the neighbours to the search queue
    search_moves.extend(next_steps)

    # While there are nodes in the search queue - there are places to visited
    while (len(search_moves) != 0):
        # Pop the next node from the search queue - queue pop
        next_node = search_moves.popleft()

        # Move the agent to the next node and colour the path
        state, reward, done = move_agent(maze, next_node, "gold", is_render)

        visited_nodes.append(maze.maze_view.robot)

        # Get a queue of valid neighbours the agent can visit from the current node
        next_steps = check_neighbour(maze, visited_nodes, search_moves, path)

        # If there are valid neighbours
        if (len(next_steps) != 0):
            # Add them to the search queue
            search_moves.extend(next_steps)

        # If the current node is the goal, then the maze is solved
        # Set solved status as True
        # Break out of the search loop
        if (done):
            solved = True
            break

    memory_footprint = sys.getsizeof(visited_nodes) + sys.getsizeof(search_moves) + sys.getsizeof(path)

    # Return the solved status and the final path
    # In theory, if there is a solution to the given maze, will always return true
    return solved, path, visited_nodes, memory_footprint


#! A* Algorithm
# Start with a closed list (visited nodes) and an open list (to be visited nodes) (priority queue for f value)
# Add the start cell to the open list (f, node coordinates)
# While there are nodes in the open list:
    # Pop a node off the open list
    # Move to it
    # Add it to closed list
    # Look at valid neighbours
    # If the neighbour is valid:
    # If the neighbour is the goal, finish the search
    # Else:
    # Calculate the new f score
    # (g = cost of current path from start, which is running total +=1)
    # (h = manhattan distance from current node to goal)
    # (f = g + h)
    # If the node is not in the open list:
    # Add the node to the open list (f, node coordinates)
def a_star(maze: gym.MazeEnv, is_render=True):
    # Solve state boolean
    solved = False

    # Visited queue - nodes the agent has already visited
    visited_nodes = deque()
    # Add the start node to the visited queue
    visited_nodes.append(maze.maze_view.entrance)

    # Path dictionary - determined path stored through node linking
    path = {}

    # Initialize the g value as 0 at start
    running_g = 0
    
    # Search queue - nodes the agent is planning on visiting
    search_nodes = []
    # Counter object - used as a tiebreaker when adding nodes to the search queue
    tiebreaker = count()
    # Heapify the search queue and add start to the priority queue
    heapq.heappush(search_nodes, (0,  next(tiebreaker),
                   maze.maze_view.entrance, running_g))

    # While there are nodes in the search priority queue - there are places to visited
    while (len(search_nodes) > 0):
        # Pop the next node from the search priority queue - priority queue pop
        next_node = heapq.heappop(search_nodes)

        # Move the agent to the next node and colour the path
        state, reward, done = move_agent(maze, next_node[2], "cyan", is_render)

        # Increment g after moving the agent - the path cost has increase by 1
        running_g = next_node[3] + 1

        # If the current node is the goal, then the maze is solved
        # Set solved status as True
        # Break out of the search loop
        if (done):
            solved = True
            break

        # Add the visited node to the visited queue
        visited_nodes.append(next_node[2])

        # For every direction available (N, E, S, W)
        for dir in maze.ACTION:
            # Calculate the neighbour given a direction
            future_node = np.copy(
                next_node[2] + np.array(maze.maze_view.maze.COMPASS[dir]))
            # If the neighbour has not been visited and there is no wall between the agent's node and the neighbour
            if (not any((future_node == elem).all() for elem in visited_nodes)) and (maze.maze_view.maze.is_open(next_node[2], dir)):
                # If the neighbour is the goal
                if (future_node == maze.maze_view.goal).all():
                    # Link the future node to the current node for plotting the final path
                    path[tuple(future_node)] = np.copy(maze.maze_view.robot)
                    # Move the agent to the neighbour and break the loop - maze solved
                    state, reward, done = move_agent(
                        maze, maze.maze_view.goal, "cyan", is_render)
                    visited_nodes.append(np.copy(maze.maze_view.robot))
                    break
                # Else - the neighbour is not the goal
                else:
                    # Calculate the new f score using the current g and the current node
                    new_f = calculate_f(maze, running_g, future_node)
                    # If the neighbour is not already in the search priority queue
                    if (not any((future_node == elem[2]).all() for elem in search_nodes)):
                        # Push the neighbour onto the priority queue
                        # Include their f score, the tiebreaker, the node and the g score
                        heapq.heappush(search_nodes, (new_f,  next(
                            tiebreaker), future_node, running_g))

                        # Link the future node to the current node for plotting the final path
                        path[tuple(future_node)] = np.copy(maze.maze_view.robot)

        # If the current node is the goal, then the maze is solved
        # Set solved status as True
        # Break out of the search loop
        if (done):
            solved = True
            break

    memory_footprint = sys.getsizeof(visited_nodes) + sys.getsizeof(search_nodes) + sys.getsizeof(path)

    # Return the solved status and final path
    # In theory, if there is a solution to the given maze, will always return true
    return solved, path, visited_nodes, memory_footprint
