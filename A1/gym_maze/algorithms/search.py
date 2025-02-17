from collections import deque
import itertools
import maze_env as mze
import numpy as np
import time


#! Check Neighbours
def check_neighbour(maze: mze.MazeEnv, visited: deque):
    next_steps = deque()

    for dir in maze.ACTION:
        future_state = (maze.maze_view.robot + np.array(maze.maze_view.maze.COMPASS[dir]))
        if (not any((future_state == elem).all() for elem in visited)) and (maze.maze_view.maze.is_open(maze.maze_view.robot, dir)):
            next_steps.append(future_state)

    return next_steps


#! DFS Algorithm - LIFO
# Start at beginning point
# Check if any neighbours are valid to move to
# Add valid neighbours to stack
# Pop entry off stack and move to it
# Repeat
def depth_first_search(maze: mze.MazeEnv):
    # Solve state
    solved = False

    # Nodes the agent has visited
    visited_nodes = deque()
    visited_nodes.append(maze.state)

    # Moves the agent needs to make to search (move to)
    search_moves = deque()
    # Get the next available moves for the agent
    next_steps = check_neighbour(maze, visited_nodes)
    # Add them to search moves
    search_moves.extend(next_steps)

    while (len(search_moves) != 0):
        next_node = search_moves.pop()

        state, reward, done, info = maze.dash(next_node)
        maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
        maze.render()

        visited_nodes.append(maze.state)

        # Get the next available moves for the agent
        next_steps = check_neighbour(maze, visited_nodes)

        if (len(next_steps) != 0):
            search_moves.extend(next_steps)

        # If the agent reaches the goal, set maze as solved and break action loop
        if (done):
            solved = True
            break

    return solved


#! BFS Algorithm - FIFO
# Start at beginning point
# Check if any neighbours are valid to move to
# Add valid neighbours to queue
# Pop entry off queue and move to it
# Repeat
def breadth_first_search(maze: mze.MazeEnv):
    # Solve state
    solved = False

    # Nodes the agent has visited
    visited_nodes = deque()
    visited_nodes.append(maze.state)

    # Moves the agent needs to make to search (move to)
    search_moves = deque()
    # Get the next available moves for the agent
    next_steps = check_neighbour(maze, visited_nodes)
    # Add them to search moves
    search_moves.extend(next_steps)

    while (len(search_moves) != 0):
        next_node = search_moves.popleft()

        state, reward, done, info = maze.dash(next_node)
        maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
        maze.render()

        visited_nodes.append(maze.state)

        # Get the next available moves for the agent
        next_steps = check_neighbour(maze, visited_nodes)

        if (len(next_steps) != 0):
            search_moves.extend(next_steps)

        # If the agent reaches the goal, set maze as solved and break action loop
        if (done):
            solved = True
            break

    return solved


#! A* Algorithm
def a_star(maze: mze.MazeEnv):
    # Solve state
    solved = False

    return solved