import random
import ttt.gym_tictactoe.env as ttt_env
import ttt.gym_tictactoe.ttt_utils as ttt_utils


# * Note: state = ((board_state), original_mark, original_player)


#! Minimax
def minimax(state, maxPlayer, curr_agent):
    
    # If this is the maximizer, set the best_score as a high negative value
    if maxPlayer:
        best_score = [-1, -999]

    # Else, this is the minimizer, set the best_score as a high positive value
    else:
        best_score = [-1, 999]

    # If the game has reached a termination state, compute the state score using the board, the original player and it's mark
    if (ttt_env.check_game_status(state[0]) >= 0):
        return ttt_utils.score_calc(state, ttt_env.check_game_status(state[0]))

    # For every available action in the board
    for action in ttt_utils.get_valid_actions(state[0]):

        # Update the board using the given action and the current agent (mark)
        state = ttt_utils.take_action(state, action, curr_agent)

        # Call minimax again, switching the current agent and player, and give the newly updated board
        score = minimax(state, not maxPlayer, ttt_env.next_mark(curr_agent))

        # Undo the given action on the board for the next action to take effect
        state = ttt_utils.undo_action(state, action)

        # Update the returned score with the action that yielded it
        score[0] = action

        # If this is the maximizer, and if the score is bigger than best, its the new best
        if maxPlayer:
            if score[1] > best_score[1]:
                # Set the best score as the current score
                best_score = score

        # Else, this is the minimizer, and if the score smaller than best, its the new best 
        else:
            if score[1] < best_score[1]:
                best_score = score

    # Return the best score
    return best_score


#! Minimax Alpha Beta Prune
def minimax_alpha_beta_prune(state, max_player, curr_agent, alpha, beta):
    
    # If this is the maximizer, set the best_score as a high negative value
    if max_player:
        best_score = [-1, -999]
    
    # Else, this is the minimizer, set the best_score as a high positive value
    else:
        best_score = [-1, 999]

    # If the game has reached a termination state, compute the state score using the board, the original player and it's mark
    if (ttt_env.check_game_status(state[0]) >= 0):
        return ttt_utils.score_calc(state, ttt_env.check_game_status(state[0]))
    
    # For every available action in the board
    for action in ttt_utils.get_valid_actions(state[0]):

        # Update the board using the given action and the current agent (mark)
        state = ttt_utils.take_action(state, action, curr_agent)
            
        # Call minimax again, switching the current agent and player, and give the newly updated board, a reduced depth and alpha, beta
        score = minimax_alpha_beta_prune(state, not max_player, ttt_env.next_mark(curr_agent), alpha, beta)
        
        # Undo the given action on the board for the next action to take effect
        state = ttt_utils.undo_action(state, action)
        
        # Update the returned score with the action that yielded it
        score[0] = action

        # If this is the maximizer
        if max_player:
            # The best is the biggest of the best and the score
            best_score = ttt_utils.max_score(best_score, score)
            
            # Alpha is the biggest of the previous alpha and the score
            alpha = max(alpha, score[1])

            # If beta is less than alpha, break out and prune the branch
            if beta < alpha:
                break

        # Else, this is the minimizer
        else:
            # The best is the smallest of the best and the score
            best_score = ttt_utils.min_score(best_score, score)

            # Beta is the smallest of the previous beta and the score
            beta = min(beta, score[1])

            # If beta is less than alpha, break out and prune the branch
            if beta < alpha:
                break
    
    # Return the best score
    return best_score


#! Qlearn Act
def qlearnAct(state, qtable, epsilon=0.4):
    ava_actions = ttt_utils.get_valid_actions(state[0])

    if random.random() < epsilon:
        best_action = random.choice(ava_actions)
    else:
        best_action = ttt_utils.find_best_action(state, ava_actions, qtable)

    return best_action, epsilon - epsilon/50000


#! Qlearn Update
def qlearnUpdate(qtable, prev_state, next_state, transition_action, score):
    lr = 0.5
    discount = 0.9
    qvalues = []

    if next_state:
        ava_actions = ttt_utils.get_valid_actions(next_state[0])
        for action in ava_actions:
            qvalues.append(qtable[action][next_state[0]])
        qtable[transition_action][prev_state[0]] += lr*(score + discount*max(qvalues) - qtable[transition_action][prev_state[0]])

    else:
        qtable[transition_action][prev_state[0]] += lr*(score - qtable[transition_action][prev_state[0]])

    return qtable
