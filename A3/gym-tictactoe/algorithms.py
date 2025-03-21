import gym_tictactoe.env as ttt_env
import utils


# * Note: state = ((board_state), mark)


#! Minimax
# Minimax Algorithm - pre-calculates all possible final states from the current board state, and determines the best action to make to ensure victory
# Works optimally when the opponent is playing optimally (max agent vs. min agent)
def minimax(state, maxPlayer, curr_agent, depth):
    # If the agent is playing as maximizier
    if maxPlayer:
        # Set the best score as incredibly small (-inf)
        best_score = [-1, -999]
    # Else, the agent must be playing as minimizer
    else:
        # Set the best score as incredibly big (+inf)
        best_score = [-1, 999]

    # If the game has reached a termination state
    if (ttt_env.check_game_status(state[0]) >= 0):
        # Calculate the game state score, depending on the agent, what the agent is playing as and the game state
        score = [-1, utils.score_calc(state, ttt_env.check_game_status(state[0]))]
        # Return the score
        return score

    # For every available action given the board
    for action in utils.get_valid_actions(state[0]):
        # Update the board using the given action and current agent
        state = utils.take_action(state, action, curr_agent)

        # Call minimax again, switching the current agent and giving the newly updated board
        score = minimax(state, not maxPlayer, ttt_env.next_mark(curr_agent), depth-1)
        # Undo the given action on the board for the next action to take effect
        state = utils.undo_action(state, action)
        # Update the returned score with the action that yielded it
        score[0] = action

        # If the agent is playing as maximizer
        if maxPlayer:
            # If the current score is greater than the best score
            if score[1] > best_score[1]:
                # Set the best score as the current score
                best_score = score
        # Else the agent must be playing as minimizer
        else:
            # If the current score is lesser than the best score
            if score[1] < best_score[1]:
                # Set the best score as the current score
                best_score = score

    # Return the best score
    return best_score


#! Minimax Alpha Beta Prune
# Minimax Algorithm w/ Alpha Beta Pruning - pre-calculates final states but prunes branches at a certain depth if they will never be visited 
# Determines the best action to make to ensure victory
# Works optimally when the opponent is playing optimally (max agent vs. min agent)
def minimax_alpha_beta_prune(state, maxPlayer, curr_agent, depth, alpha, beta):
    # If the agent is playing as maximizier
    if maxPlayer:
        # Set the best score as incredibly small (-inf)
        best_score = [-1, -999]
    # Else, the agent must be playing as minimizer
    else:
        # Set the best score as incredibly big (+inf)
        best_score = [-1, 999]

    # If the game has reached a termination state
    if (ttt_env.check_game_status(state[0]) >= 0):
        # Calculate the game state score, depending on the agent, what the agent is playing as and the game state
        score = [-1, utils.score_calc(state, ttt_env.check_game_status(state[0]))]
        # Return the score
        return score
    
    # For every available action given the board
    for action in utils.get_valid_actions(state[0]):
        # Update the board using the given action and current agent
        state = utils.take_action(state, action, curr_agent)
            
        # Call minimax again, switching the current agent and giving the newly updated board
        score = minimax_alpha_beta_prune(state, not maxPlayer, ttt_env.next_mark(curr_agent), depth-1, alpha, beta)
        # Undo the given action on the board for the next action to take effect
        state = utils.undo_action(state, action)
        
        # Update the returned score with the action that yielded it
        score[0] = action

        # If the agent is playing as maximizer
        if maxPlayer:
            best_score = utils.max_score(best_score, score)
            alpha = max(alpha, score[1])
            # If the current score is greater than the best score
            if beta < alpha:
                break

        # Else the agent must be playing as minimizer
        else:
            best_score = utils.min_score(best_score, score)
            beta = min(beta, score[1])
            if beta < alpha:
                break
    
    # Return the best score
    return best_score
