import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, unique
from operator import itemgetter
from typing import Tuple, NamedTuple, Hashable, Optional
import time

import gym
import numpy as np
import pygame
from gym import error
from gym import spaces

from c4.gym_connect_four.envs.render import render_board


@unique
class ResultType(Enum):
    NONE = None
    DRAW = 0
    WIN1 = 1
    WIN2 = -1

    def __eq__(self, other):
        """
        Need to implement this due to an unfixed bug in Python since 2017: https://bugs.python.org/issue30545
        """
        return self.value == other.value


class ConnectFourEnv(gym.Env):
    """
    Description:
        ConnectFour game environment

    Observation:
        Type: Discreet(6,7)

    Actions:
        Type: Discreet(7)
        Num     Action
        x       Column in which to insert next token (0-6)

    Reward:
        Reward is 0 for every step.
        If there are no other further steps possible, Reward is 0.5 and termination will occur
        If it's a win condition, Reward will be 1 and termination will occur
        If it is an invalid move, Reward will be -1 and termination will occur

    Starting State:
        All observations are assigned a value of 0

    Episode Termination:
        No more spaces left for pieces
        4 pieces are present in a line: horizontal, vertical or diagonally
        An attempt is made to place a piece in an invalid location
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    LOSS_REWARD = -1
    DEF_REWARD = 0
    DRAW_REWARD = 0.5
    WIN_REWARD = 1

    class StepResult(NamedTuple):

        res_type: ResultType

        def get_reward(self, player: int):
            if self.res_type is ResultType.NONE:
                return ConnectFourEnv.DEF_REWARD
            elif self.res_type is ResultType.DRAW:
                return ConnectFourEnv.DRAW_REWARD
            else:
                return {ResultType.WIN1.value: ConnectFourEnv.WIN_REWARD, ResultType.WIN2.value: ConnectFourEnv.LOSS_REWARD}[
                    self.res_type.value * player]

        def is_done(self):
            return self.res_type != ResultType.NONE

    def __init__(self, board_shape=(6, 7), window_width=512, window_height=512):
        super(ConnectFourEnv, self).__init__()

        self.board_shape = board_shape

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.__current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)

        self.__player_color = 1
        self.__screen = None
        self.__window_width = window_width
        self.__window_height = window_height
        self.__rendered_board = self._update_board_render()

    # Training Gameloop
    def train_game(self, player1, player2, board: Optional[np.ndarray] = None, render=False, start_player=1) -> ResultType:
        self.reset()

        self.__current_player = start_player

        def cp(): return self.__current_player

        def get_current_player():
            return player1 if cp() == 1 else player2

        def change_player():
            self.__current_player *= -1
            return get_current_player()

        if get_current_player().name != "QlearnPlayer":
            player = get_current_player()
            transition_action = player.get_next_action()
            step_result = self._step(transition_action)
            change_player()

        prev_board = self.board
        transition_action = get_current_player().get_next_action(True)

        score = 0

        done = False

        while not done:
            step_result = self._step(transition_action)
            reward = step_result.get_reward(cp())
            done = step_result.is_done()

            if done:
                if self.ghost_check_winner(self.board) == cp():
                    score = 1
                    break
                else:
                    score = 0
                    break

            player = change_player()

            action = player.get_next_action()
            step_result = self._step(action)
            reward = step_result.get_reward(cp())
            done = step_result.is_done()

            if done:
                if self.ghost_check_winner(self.board) == cp():
                    score = -1
                    break
                else:
                    score = 0
                    break

            player = change_player()

            new_action = player.get_next_action(True)
            next_board = self.board
            player.update(prev_board, next_board, transition_action, score)
            prev_board = next_board
            transition_action = new_action

        player1.update(prev_board, None, transition_action, score)
        reward = step_result.get_reward(cp())

        return step_result.res_type

    # Run Gameloop
    def run_game(self, player1, player2, board: Optional[np.ndarray] = None, render=False, start_player=1) -> ResultType:
        self.reset()

        self.__current_player = start_player

        def cp(): return self.__current_player

        def get_current_player():
            return player1 if cp() == 1 else player2

        def change_player():
            self.__current_player *= -1
            return get_current_player()
        
        p1_actions = 0

        player = get_current_player()
        act = player.get_next_action()
        step_result = self._step(act)

        if get_current_player().player == 1:
                p1_actions += 1

        player = change_player()

        done = False

        while not done:
            if render:
                self.render()
                time.sleep(1)
            step_result = self._step(player.get_next_action())

            if get_current_player().player == 1:
                p1_actions += 1

            player = change_player()

            reward = step_result.get_reward(cp())
            done = step_result.is_done()

        reward = step_result.get_reward(cp())
        if render:
            self.render()
            time.sleep(5)

        return step_result.res_type, p1_actions

    def run(self, player1, player2, board: Optional[np.ndarray] = None, render=False) -> ResultType:
        # player1.reset()
        # player2.reset()
        self.reset(board)

        def cp(): return self.__current_player

        def change_player():
            self.__current_player *= -1
            return player1 if cp() == 1 else player2

        # state_hist = deque([self.__board.copy()], maxlen=4)

        act = player1.get_next_action(self.__board * 1)
        # act_hist = deque([act], maxlen=2)
        step_result = self._step(act)
        # state_hist.append(self.__board.copy())
        player = change_player()
        done = False
        while not done:
            if render:
                self.render()
            # act_hist.append(player.get_next_action(self.__board * cp()))
            step_result = self._step(
                player.get_next_action(self.__board * cp()))
            # step_result = self._step(act_hist[-1])
            # state_hist.append(self.__board.copy())

            player = change_player()

            reward = step_result.get_reward(cp())
            done = step_result.is_done()
            # player.learn(state=state_hist[-3] * cp(), action=act_hist[-2], state_next=state_hist[-1] * cp(), reward=reward, done=done)

        player = change_player()
        reward = step_result.get_reward(cp())
        # player.learn(state_hist[-2] * cp(), act_hist[-1], state_hist[-1] * cp(), reward, done)
        if render:
            self.render()

        return step_result.res_type

    # Take a ghost step on a copy of a board
    def ghost_step(self, board, action, player):
        for index in list(reversed(range(self.board_shape[0]))):
            if board[index][action] == 0:
                board[index][action] = player
                return board.copy()

    # Returns a tuple of board, reward, done, dict
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        step_result = self._step(action)
        reward = step_result.get_reward(self.__current_player)
        done = step_result.is_done()
        return self.__board.copy(), reward, done, {}

    def _step(self, action: int) -> StepResult:
        result = ResultType.NONE

        if not self.is_valid_action(action):
            raise Exception(
                'Unable to determine a valid move! Maybe invoke at the wrong time?'
            )

        # Check and perform action
        for index in list(reversed(range(self.board_shape[0]))):
            if self.__board[index][action] == 0:
                self.__board[index][action] = self.__current_player
                break

        # Check if board is completely filled
        if np.count_nonzero(self.__board[0]) == self.board_shape[1]:
            result = ResultType.DRAW
        else:
            # Check win condition
            if self.is_win_state():
                result = ResultType.WIN1 if self.__current_player == 1 else ResultType.WIN2
        return self.StepResult(result)

    @property
    def board(self):
        return self.__board.copy()

    def reset(self, board: Optional[np.ndarray] = None) -> np.ndarray:
        self.__current_player = 1
        if board is None:
            self.__board = np.zeros(self.board_shape, dtype=int)
        else:
            self.__board = board
        self.__rendered_board = self._update_board_render()
        return self.board

    def render(self, mode: str = 'console', close: bool = False) -> None:
        if mode == 'console':
            replacements = {
                self.__player_color: 'A',
                0: ' ',
                -1 * self.__player_color: 'B'
            }

            def render_line(line):
                return "|" + "|".join(
                    ["{:>2} ".format(replacements[x]) for x in line]) + "|"

            hline = '|---+---+---+---+---+---+---|'
            print(hline)
            for line in np.apply_along_axis(render_line,
                                            axis=1,
                                            arr=self.__board):
                print(line)
            print(hline)

        elif mode == 'human':
            if self.__screen is None:
                pygame.init()
                self.__screen = pygame.display.set_mode(
                    (round(self.__window_width), round(self.__window_height)))

            if close:
                pygame.quit()

            self.__rendered_board = self._update_board_render()
            frame = self.__rendered_board
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.rotate(surface, 90)
            self.__screen.blit(surface, (0, 0))

            pygame.display.update()
        elif mode == 'rgb_array':
            self.__rendered_board = self._update_board_render()
            frame = self.__rendered_board
            return np.flip(frame, axis=(0, 1))
        else:
            raise error.UnsupportedMode()

    def close(self) -> None:
        pygame.quit()
        self.__screen = None

    def is_valid_action(self, action: int) -> bool:
        return self.__board[0][action] == 0

    def _update_board_render(self) -> np.ndarray:
        return render_board(self.__board,
                            image_width=self.__window_width,
                            image_height=self.__window_height)

    # Calculate a score heuristically using a ghost board
    def ghost_heuristic(self, board, original_player) -> int:
        score = 0

        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(board[i][j:j + 4])
                if value == 3:
                    if original_player[0] == 1:
                        if original_player[1]:
                            score += 2
                        else:
                            score -= 2

                    else:
                        if original_player[1]:
                            score -= 1
                        else:
                            score += 1
                elif value == -3:
                    if original_player[0] == -1:
                        if original_player[1]:
                            score += 2
                        else:
                            score -= 2

                    else:
                        if original_player[1]:
                            score -= 1
                        else:
                            score += 1

                if value == 2:
                    if original_player[0] == 1:
                        if original_player[1]:
                            score += 1
                        else:
                            score -= 1

                    else:
                        if original_player[1]:
                            score -= 0.5
                        else:
                            score += 0.5
                elif value == -2:
                    if original_player[0] == -1:
                        if original_player[1]:
                            score += 1
                        else:
                            score -= 1

                    else:
                        if original_player[1]:
                            score -= 0.5
                        else:
                            score += 0.5

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if value == 3:
                    if original_player[0] == 1:
                        if original_player[1]:
                            score += 2
                        else:
                            score -= 2

                    else:
                        if original_player[1]:
                            score -= 1
                        else:
                            score += 1
                elif value == -3:
                    if original_player[0] == -1:
                        if original_player[1]:
                            score += 2
                        else:
                            score -= 2

                    else:
                        if original_player[1]:
                            score -= 1
                        else:
                            score += 1

                if value == 2:
                    if original_player[0] == 1:
                        if original_player[1]:
                            score += 1
                        else:
                            score -= 1

                    else:
                        if original_player[1]:
                            score -= 0.5
                        else:
                            score += 0.5
                elif value == -2:
                    if original_player[0] == -1:
                        if original_player[1]:
                            score += 1
                        else:
                            score -= 1

                    else:
                        if original_player[1]:
                            score -= 0.5
                        else:
                            score += 0.5

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += board[i + k][j + k]
                    if value == 3:
                        if original_player[0] == 1:
                            if original_player[1]:
                                score += 2
                            else:
                                score -= 2

                        else:
                            if original_player[1]:
                                score -= 1
                            else:
                                score += 1
                    elif value == -3:
                        if original_player[0] == -1:
                            if original_player[1]:
                                score += 2
                            else:
                                score -= 2

                        else:
                            if original_player[1]:
                                score -= 1
                            else:
                                score += 1

                    if value == 2:
                        if original_player[0] == 1:
                            if original_player[1]:
                                score += 1
                            else:
                                score -= 1

                        else:
                            if original_player[1]:
                                score -= 0.5
                            else:
                                score += 0.5
                    elif value == -2:
                        if original_player[0] == -1:
                            if original_player[1]:
                                score += 1
                            else:
                                score -= 1

                        else:
                            if original_player[1]:
                                score -= 0.5
                            else:
                                score += 0.5

        reversed_board = np.fliplr(board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if value == 3:
                        if original_player[0] == 1:
                            if original_player[1]:
                                score += 2
                            else:
                                score -= 2

                        else:
                            if original_player[1]:
                                score -= 1
                            else:
                                score += 1
                    if value == -3:
                        if original_player[0] == -1:
                            if original_player[1]:
                                score += 2
                            else:
                                score -= 2

                        else:
                            if original_player[1]:
                                score -= 1
                            else:
                                score += 1

                    if value == 2:
                        if original_player[0] == 1:
                            if original_player[1]:
                                score += 1
                            else:
                                score -= 1

                        else:
                            if original_player[1]:
                                score -= 0.5
                            else:
                                score += 0.5
                    elif value == -2:
                        if original_player[0] == -1:
                            if original_player[1]:
                                score += 1
                            else:
                                score -= 1

                        else:
                            if original_player[1]:
                                score -= 0.5
                            else:
                                score += 0.5

        return score

    # Check the winner on a ghost board
    def ghost_check_winner(self, board) -> int:
        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(board[i][j:j + 4])
                if value == 4:
                    return 1
                elif value == -4:
                    return -1

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if value == 4:
                    return 1
                elif value == -4:
                    return -1

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += board[i + k][j + k]
                    if value == 4:
                        return 1
                    elif value == -4:
                        return -1

        reversed_board = np.fliplr(board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if value == 4:
                        return 1
                    elif value == -4:
                        return -1

        if np.count_nonzero(board[0]) == self.board_shape[1]:
            return 0

    # Check the board is terminal with a ghost board
    def ghost_is_terminal_state(self, board) -> bool:
        if np.count_nonzero(board[0]) == self.board_shape[1]:
            return True

        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def is_win_state(self) -> bool:
        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(self.__board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*self.__board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += self.__board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(self.__board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    # Get the available actions on a ghost board
    def ghost_available_moves(self, board) -> frozenset:
        return frozenset(
            (i for i in range(self.board_shape[1]) if board[0][i] == 0))

    def available_moves(self) -> frozenset:
        return frozenset(
            (i for i in range(self.board_shape[1]) if self.is_valid_action(i)))
