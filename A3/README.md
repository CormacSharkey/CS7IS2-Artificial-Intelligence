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