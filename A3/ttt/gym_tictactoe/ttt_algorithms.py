import random
import ttt.gym_tictactoe.env as ttt_env
import ttt.gym_tictactoe.ttt_utils as ttt_utils


# * Note: state = ((board_state), original_mark, original_player)


#! Minimax Algorithm
def minimax(state, max_player, curr_agent):
    # Best score depends on max_player
    if max_player:
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # If the game is over, return the state score
    if (ttt_env.check_game_status(state[0]) >= 0):
        return ttt_utils.score_calc(state, ttt_env.check_game_status(state[0]))

    # Take every available action as the current agent, then recursively use minimax again
    for action in ttt_utils.get_valid_actions(state[0]):

        # Take an action and call minimax
        state = ttt_utils.take_action(state, action, curr_agent)
        score = minimax(state, not max_player, ttt_env.next_mark(curr_agent))

        # Update the returned score
        score[0] = action

        # Updating the best score depends on max_player
        if max_player:
            if score[1] > best_score[1]:
                best_score = score
        else:
            if score[1] < best_score[1]:
                best_score = score

    # Return the best score
    return best_score


#! Minimax Alpha Beta Prune Algorithm
def minimax_alpha_beta_prune(state, max_player, curr_agent, alpha, beta):
    # Best score depends on max_player
    if max_player:
        best_score = [-1, -999]
    else:
        best_score = [-1, 999]

    # If the game is over, return the state score
    if (ttt_env.check_game_status(state[0]) >= 0):
        return ttt_utils.score_calc(state, ttt_env.check_game_status(state[0]))
    
    # Take every available action as the current agent, then recursively use minimax again
    for action in ttt_utils.get_valid_actions(state[0]):

        # Take an action and call minimax prune
        state = ttt_utils.take_action(state, action, curr_agent)
        score = minimax_alpha_beta_prune(state, not max_player, ttt_env.next_mark(curr_agent), alpha, beta)
        
        # Update the returned score
        score[0] = action

        # Updating the best score depends on max_player
        if max_player:
            # Get the best score and compare it to alpha
            best_score = ttt_utils.max_score(best_score, score)
            alpha = max(alpha, score[1])
            if beta < alpha:
                break
        else:
            # Get the best score and compare it to beta
            best_score = ttt_utils.min_score(best_score, score)
            beta = min(beta, score[1])
            if beta < alpha:
                break
    
    # Return the best score
    return best_score


#! Qlearn Act Algorithm
def qlearnAct(state, qtable, epsilon=0.4):
    # Get the available actions
    ava_actions = ttt_utils.get_valid_actions(state[0])

    # Exploration vs. Exploitation - random exploration vs. Qtable decision
    if random.random() < epsilon:
        best_action = random.choice(ava_actions)
    else:
        best_action = ttt_utils.find_best_action(state, ava_actions, qtable)

    # Return the best action and decayed epsilon
    return best_action, epsilon - epsilon/50000


#! Qlearn Update Algorithm
def qlearnUpdate(qtable, prev_state, next_state, transition_action, score):
    # Declare parameters
    lr = 0.5
    discount = 0.9
    qvalues = []

    # If this isn't the terminal update, update with next_state
    if next_state:
        ava_actions = ttt_utils.get_valid_actions(next_state[0])
        for action in ava_actions:
            qvalues.append(qtable[action][next_state[0]])
        qtable[transition_action][prev_state[0]] += lr*(score + discount*max(qvalues) - qtable[transition_action][prev_state[0]])
    else:
        qtable[transition_action][prev_state[0]] += lr*(score - qtable[transition_action][prev_state[0]])

    # Return the Qtable
    return qtable
