from collections import deque
import itertools
import maze_env as gym
import numpy as np
import time
import heapq
from itertools import count


#! Check Neighbours
def check_neighbour(maze: gym.MazeEnv, visited: deque):
    next_steps = deque()

    for dir in maze.ACTION:
        future_state = (maze.maze_view.robot + np.array(maze.maze_view.maze.COMPASS[dir]))
        if (not any((future_state == elem).all() for elem in visited)) and (maze.maze_view.maze.is_open(maze.maze_view.robot, dir)):
            next_steps.append(future_state)

    return next_steps

#! Calculate F
def calculate_f(maze: gym.MazeEnv, running_g, future_node):
    new_g = running_g
    new_h = np.sum(np.abs(maze.maze_view.goal - future_node)) # Manhattan Distance
    # new_h = int(np.linalg.norm(maze.maze_view.goal - future_node)) # Euclidean Distance
    new_f = new_g + new_h

    return new_f

def move_agent(maze: gym.MazeEnv, node, colour):
    state, reward, done, info = maze.dash(node)
    if (colour == "purple"):
        maze.maze_view._MazeView2D__colour_explored_cell(state, (180, 0, 180), 180)
    elif (colour == "gold"):
        maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
    elif (colour == "cyan"):
        maze.maze_view._MazeView2D__colour_explored_cell(state, (0, 139, 139), 180)
    maze.render()
    # time.sleep(2)

    return state, reward, done

#! DFS Algorithm - LIFO
# Start at beginning point
# Check if any neighbours are valid to move to
# Add valid neighbours to stack
# Pop entry off stack and move to it
# Repeat
def depth_first_search(maze: gym.MazeEnv):
    # Solve state
    solved = False

    # Nodes the agent has visited
    visited_nodes = deque()
    visited_nodes.append(maze.maze_view.entrance)

    # Moves the agent needs to make to search (move to)
    search_moves = deque()
    # Get the next available moves for the agent
    next_steps = check_neighbour(maze, visited_nodes)

    for node in next_steps:
        visited_nodes.append(node)

    # Add them to search moves
    search_moves.extend(next_steps)

    while (len(search_moves) != 0):
        next_node = search_moves.pop()

        # state, reward, done, info = maze.dash(next_node)
        # maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
        # maze.render()

        state, reward, done = move_agent(maze, next_node, "purple")

        # visited_nodes.append(next_node)

        # Get the next available moves for the agent
        next_steps = check_neighbour(maze, visited_nodes)

        if (len(next_steps) != 0):
            search_moves.extend(next_steps)
            for node in next_steps:
                visited_nodes.append(node)

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
def breadth_first_search(maze: gym.MazeEnv):
    # Solve state
    solved = False

    # Nodes the agent has visited
    visited_nodes = deque()
    visited_nodes.append(maze.maze_view.entrance)

    # Moves the agent needs to make to search (move to)
    search_moves = deque()
    # Get the next available moves for the agent
    next_steps = check_neighbour(maze, visited_nodes)

    for node in next_steps:
        visited_nodes.append(node)

    # Add them to search moves
    search_moves.extend(next_steps)

    while (len(search_moves) != 0):
        next_node = search_moves.popleft()

        # state, reward, done, info = maze.dash(next_node)
        # maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
        # maze.render()
        state, reward, done = move_agent(maze, next_node, "gold")

        # visited_nodes.append(next_node)

        # Get the next available moves for the agent
        next_steps = check_neighbour(maze, visited_nodes)

        if (len(next_steps) != 0):
            search_moves.extend(next_steps)
            for node in next_steps:
                visited_nodes.append(node)

        # If the agent reaches the goal, set maze as solved and break action loop
        if (done):
            solved = True
            break

    return solved


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
            # Calculate the new f score (g = cost of current path from start, which is running total +=1) (h = manhattan distance from current node to goal) (f = g + h)
            # If the node is not in the open list (or "new f value is smaller"):
                # Add the node to the open list (f, node coordinates)
def a_star(maze: gym.MazeEnv):
        # Solve state
    solved = False

    visited_nodes = deque()
    visited_nodes.append(maze.maze_view.entrance)

    search_nodes = []
    tiebreaker = count()
    heapq.heappush(search_nodes, (0,  next(tiebreaker), maze.maze_view.entrance, 0))

    running_g = 0

    while (len(search_nodes) > 0):
        next_node = heapq.heappop(search_nodes)
        
        # state, reward, done, info = maze.dash(next_node[2])
        # maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
        # maze.render()
        state, reward, done = move_agent(maze, next_node[2], "cyan")

        running_g = next_node[3] + 1

        if (done):
            solved = True
            break

        visited_nodes.append(next_node[2])

        for dir in maze.ACTION:
            future_node = (next_node[2] + np.array(maze.maze_view.maze.COMPASS[dir]))
            if (not any((future_node == elem).all() for elem in visited_nodes)) and (maze.maze_view.maze.is_open(next_node[2], dir)):
                if (future_node == maze.maze_view.goal).all():
                    # state, reward, done, info = maze.dash(maze.maze_view.goal)
                    # maze.maze_view._MazeView2D__colour_explored_cell(state, (187, 165, 61), 180)
                    # maze.render()
                    state, reward, done = move_agent(maze, maze.maze_view.goal, "cyan")
                    break
                else:
                    new_f = calculate_f(maze, running_g, future_node)
                    if (not any((future_node == elem[2]).all() for elem in search_nodes)):
                        heapq.heappush(search_nodes, (new_f,  next(tiebreaker), future_node, running_g))

        if (done):
            solved = True
            break

    return solved