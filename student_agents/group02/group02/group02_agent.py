import random
from typing import Tuple, List

import numpy as np

from group02 import util
from pommerman import agents
from pommerman.constants import Action, Item


class Group02Agent(agents.BaseAgent):
    """
    This is the class of your agent. During the tournament an object of this class
    will be created for every game your agents plays.
    If you exceed 500 MB of main memory used, your agent will crash.

    Args:
        ignore the arguments passed to the constructor
        in the constructor you can do initialisation that must be done before a game starts
    """
    def __init__(self, *args, **kwargs):
        super(Group02Agent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        """
        Every time your agent is required to send a move, this method will be called.
        You have 0.5 seconds to return a move, otherwise no move will be played.

        Parameters
        ----------
        obs: dict
            keys:
                'alive': {list:2}, board ids of agents alive
                'board': {ndarray: (11, 11)}, board representation
                'bomb_blast_strength': {ndarray: (11, 11)}, describes range of bombs
                'bomb_life': {ndarray: (11, 11)}, shows ticks until bomb explodes
                'bomb_moving_direction': {ndarray: (11, 11)}, describes moving direction if bomb has been kicked
                'flame_life': {ndarray: (11, 11)}, ticks until flame disappears
                'game_type': {int}, irrelevant for you, we only play FFA version
                'game_env': {str}, irrelevant for you, we only use v0 env
                'position': {tuple: 2}, position of the agent (row, col)
                'blast_strength': {int}, range of own bombs         --|
                'can_kick': {bool}, ability to kick bombs             | -> can be improved by collecting items
                'ammo': {int}, amount of bombs that can be placed   --|
                'teammate': {Item}, irrelevant for you
                'enemies': {list:3}, possible ids of enemies, you only have one enemy in a game!
                'step_count': {int}, if 800 steps were played then game ends in a draw (no points)

        action_space: spaces.Discrete(6)
            action_space.sample() returns a random move (int)
            6 possible actions in pommerman (integers 0-5)

        Returns
        -------
        action: int
            Stop (0): This action is a pass.
            Up (1): Move up on the board.
            Down (2): Move down on the board.
            Left (3): Move left on the board.
            Right (4): Move right on the board.
            Bomb (5): Lay a bomb.
        """
        my_position: Tuple[int, int] = tuple(obs['position'])
        board: np.ndarray = np.array(obs['board'])
        bomb_blast_strength: np.ndarray = np.array(obs['bomb_blast_strength'])
        bomb_life: np.ndarray = np.array(obs['bomb_life'])
        enemy: Item = obs['enemies'][0]  # we only have to deal with 1 enemy
        epos: np.ndarray = np.where(obs['board'] == enemy.value)
        enemy_position: Tuple[int, int] = (epos[0][0], epos[1][0])
        ammo: int = int(obs['ammo'])
        blast_strength: int = int(obs['blast_strength'])

        actions = self.legal_moves(my_position, board, bomb_blast_strength[my_position[0], my_position[1]] != 0, ammo)

        return random.choice(actions)

    def legal_moves(self, position: Tuple[int, int], board: np.ndarray, on_bomb: bool, ammo: int) -> List[int]:
        """
        Filters actions like bumping into a wall (which is equal to "Stop" action) or trying
        to lay a bomb, although there is no ammo available
        """
        all_actions = [Action.Stop.value]  # always possible
        if not on_bomb and ammo > 0:
            all_actions.append(Action.Bomb.value)

        up = position[0] - 1
        down = position[0] + 1
        left = position[1] - 1
        right = position[1] + 1

        if up >= 0 and self.is_accessible(board[up, position[1]]):
            all_actions.append(Action.Up.value)
        if down < len(board) and self.is_accessible(board[down, position[1]]):
            all_actions.append(Action.Down.value)
        if left >= 0 and self.is_accessible(board[position[0], left]):
            all_actions.append(Action.Left.value)
        if right < len(board) and self.is_accessible(board[position[0], right]):
            all_actions.append(Action.Right.value)

        return all_actions

    @staticmethod
    def is_accessible(pos_val: int) -> bool:
        return pos_val in util.ACCESSIBLE_TILES
