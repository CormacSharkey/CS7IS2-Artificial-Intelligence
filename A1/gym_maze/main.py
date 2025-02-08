import maze_env as mze
import random


def main():

    maze = mze.MazeEnv(maze_size=(10, 10))
    maze.render()

    # print(maze.maze_view.robot)
    # print(maze.maze_view.goal)

    total_reward = 0

    for itx in range(10000):
        # Assign result of bfs, dfs, etc. to "action"
        action = maze.ACTION[random.randrange(4)]

        # maze.maze_view._MazeView2D__colour_cell(maze.maze_view.robot, (100, 0, 255), 200)
        state, reward, done, info = maze.step(action)
        maze.render()

        if (done):
            print("You've reached the goal!")
            break

        # Feed "total_reward" to search function
        total_reward += reward
        # print(total_reward)

    input("Enter any key to quit.")


if __name__ == "__main__":
    main()
