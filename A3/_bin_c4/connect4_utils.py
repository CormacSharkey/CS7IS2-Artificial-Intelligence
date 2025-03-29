import gym_connect4.envs.connect4_env as connect4


#! Score Calc
# Calculate the terminal state score - the win, lose or draw
def score_calc(board: connect4.Connect4, originalPlayer):
    if board.is_draw():
        return 0
    elif board.is_winner(originalPlayer[0]):
        if (originalPlayer[1]):
            return 100
        else:
            return -100
    else:
        if (originalPlayer[1]):
            return -100
        else:
            return 100


#! Evaluation Function
# Determine a score for the current board state heuristically
def evaluation_function(board: connect4.Connect4, originalPlayer):
    score = 0

    # I think this is ok
    clone_board = board.clone()
    clone_board.player = originalPlayer[0]

    # if (originalPlayer[1]):
    #     score += 3*clone_board.is_potential_winner(originalPlayer[0], 0)
    # else:
    #     score -= 3*clone_board.is_potential_winner(originalPlayer[0], 0)

    # if (originalPlayer[1]):
    #     score -= 3*clone_board.is_potential_winner(1-originalPlayer[0], 0)
    # else:
    #     score += 3*clone_board.is_potential_winner(1-originalPlayer[0], 0)

    if (originalPlayer[1]):
        score += 3*clone_board.is_potential_winner(neighbours=1)
    else:
        score -= 3*clone_board.is_potential_winner(neighbours=1)

    if (originalPlayer[1]):
        score -= 3*clone_board.is_potential_winner(clone_board.player^1, neighbours=1)
    else:
        score += 3*clone_board.is_potential_winner(clone_board.player^1, neighbours=1)

    if (originalPlayer[1]):
        score += 2*clone_board.is_potential_winner(neighbours=2)
    else:
        score -= 2*clone_board.is_potential_winner(neighbours=2)

    if (originalPlayer[1]):
        score -= 2*clone_board.is_potential_winner(clone_board.player^1, neighbours=1)
    else:
        score += 2*clone_board.is_potential_winner(clone_board.player^1, neighbours=1)
            
    if score == 0:
        if (originalPlayer[1]):
            score = -10
        else:
            score = 10

    del clone_board

    # for action in ava_actions:
    #     clone_board  = board.clone()
    #     clone_board.player = originalPlayer[0]
    #     clone_board.move(action)

    #     if (clone_board.is_potential_winner(originalPlayer[0], 0)):
    #         if (originalPlayer[1]):
    #             score += 3
    #         else:
    #             score -= 3

    #     elif (clone_board.is_potential_winner(originalPlayer[0], 1)):
    #         if (originalPlayer[1]):
    #             score += 2
    #         else:
    #             score -= 2

    #     elif (clone_board.is_potential_winner(originalPlayer[0], 2)):
    #         if (originalPlayer[1]):
    #             score += 1
    #         else:
    #             score -= 1
    #     else:
    #         if (originalPlayer[1]):
    #             score -= 0.5
    #         else:
    #             score += 0.5

    #     del clone_board

    return score