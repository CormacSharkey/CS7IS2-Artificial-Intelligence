# CS7IS2 Artificial Intelligence A1

Author: CormacSharkey

## Getting Started

### Dependencies

* Access the **gym_maze** directory

```
cd A1/gym_maze
```

* Install the dependencies with **pip**

```
pip install -r requirements.txt
```

### Executing Maze Solvers

* From the **gym_maze** directory, run **main.py** with **python**

* Two arguments may be specified:

    * **algo**: the keyname to choose an algorithm
    * **size**: the size of the maze; size x size tiles

    Note: the default **algo** runs all algorithms with 5 second intervals, and the defaut **size** is 15 x 15 tiles

Examples:
```
python main.py --algo "dfs" --size 13
python main.py --algo "bfs" 
python main.py --algo "a*" --size 17
python main.py --algo "vi"  --size 26
python main.py --algo "pi" 
python main.py --size 35
```

### Running Benchmarks
* From the **gym_maze** directory, run **benchmark.py** with **python**

* Two arguments may be specified:

    * **runs**: the no. of benchmark runs for averaging metrics
    * **size**: the size of the maze; size x size tiles 

    Note: the default **runs** value is 10, and the defaut **size** is 15 x 15 tiles

Examples:
```
python benchmark.py --runs 20 --size 20
python benchmark.py --runs 15
python benchmark.py --size 35
```

## Credits
+ Maze Env Code Repo: https://github.com/MattChanTK/gym-maze

# CS7IS2 Artificial Intelligence A3

Author: CormacSharkey

## Getting Started

### Dependencies

* Access the **A3** directory

```
cd A3
```

* Install the dependencies with **pip**

```
pip install -r requirements.txt
```

### Running Agents in Environments

* From the **A3** directory, run **main.py** with **python**

* Five arguments may be specified:

    * **game**: the game environment
    * **players**: the agents playing each other
    * **episodes**: the number of episodes to run for
    * **ttt_epochs**: the number of training epochs for ttt
    * **c4_epochs**: the number of training epochs for c4

    Note: default game value is **ttt**, default players value is **MPC**, default episodes value is **1**, default ttt_epochs value is **50000**, default c4_epochs value is **50000**

Examples:
```
python main.py --game ttt --players MC --episodes 27
python main.py --game c4 --players MPC --episodes 76
python main.py --game ttt --players QC --episodes 48 --ttt_epochs 100000
python main.py --game all --players QM --episodes 10
python main.py --game c4 --players QM --episodes 10 --c4_epochs 25000
```

## Credits
* TicTacToe Env Code Repo: https://github.com/haje01/gym-tictactoe
* Connect4 Env Code Repo: https://github.com/IASIAI/gym-connect-four