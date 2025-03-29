import gym_connect4.envs.connect4_env as connect4
import connect4_utils as utils


# Need to make depth even number to then check how many winning moves exist
def minimax_raw(board: connect4.Connect4, maxPlayer, originalPlayer, depth):
    print(f"Depth: {depth}")

    if maxPlayer:
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # if the game has ended, compute the state score
    if (board.is_game_over()):
        score = [-1, utils.score_calc(board, originalPlayer)]
        return score
    
    for action in board.get_moves():
        board.move(action)
        clone_board = board.clone()

        score = minimax_raw(clone_board, not maxPlayer, originalPlayer, depth+1)

        score[0] = action

        if maxPlayer:
            if score[1] > best_score[1]:
                best_score = score

        else:
            if score[1] < best_score[1]:
                best_score = score

    return best_score


# Need to make depth even number to then check how many winning moves exist
def minimax_heuristic(board: connect4.Connect4, maxPlayer, originalPlayer, depth):

    if (depth < 7):
        if maxPlayer:
            best_score = [-1, -999]
        else:
            best_score = [-1, 999]

        # if the game has ended, compute the state score
        if (board.is_game_over()):
            score = [-1, utils.score_calc(board, originalPlayer)]
            return score
        
        for action in board.get_moves():
            board.move(action)
            clone_board = board.clone()

            score = minimax_heuristic(clone_board, not maxPlayer, originalPlayer, depth+1)

            score[0] = action

            if maxPlayer:
                if score[1] > best_score[1]:
                    best_score = score

            else:
                if score[1] < best_score[1]:
                    best_score = score

    else:
        # if the game has ended, compute the state score
        if (board.is_game_over()):
            best_score = [-1, utils.score_calc(board, originalPlayer)]
        else:        
            best_score = [-1, utils.evaluation_function(board, originalPlayer)]

    return best_score     
        