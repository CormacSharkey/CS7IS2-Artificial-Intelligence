import maze_env as mze
import random


def main():

    maze = mze.MazeEnv(maze_size=(20, 20))

    print(maze.maze_view.robot)
    print(maze.maze_view.goal)
    
    total_reward = 0

    for itx in range(10000):
        maze.render()

        # Assign result of bfs, dfs, etc. to "action"
        action = maze.ACTION[random.randrange(4)]

        state, reward, done, info = maze.step(action)

        # Feed "total_reward" to search function
        total_reward += reward
        # print(total_reward)

    input("Enter any key to quit.")

if __name__ == "__main__":
    main()