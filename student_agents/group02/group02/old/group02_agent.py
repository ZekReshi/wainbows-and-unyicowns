import random
from typing import Tuple, List

import numpy as np
import time

from group02 import util
from group02.util import FindItemPredicate, FindWoodPredicate, WillDiePredicate, MOVE_ACTIONS, MAX_BOMB_LIFE, \
    SurvivePredicate
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
        self.enemy_can_kick = False
        self.enemy_blast_strength = 2
        self.enemy_max_ammo = 1
        self.enemy_ammo = 1
        self.enemy_bombs = list()

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
        bomb_moving_direction: np.ndarray = np.array(obs['bomb_moving_direction'])
        flame_life: np.ndarray = np.array(obs['flame_life'])
        enemy: Item = obs['enemies'][0]  # we only have to deal with 1 enemy
        epos: np.ndarray = np.where(obs['board'] == enemy.value)
        enemy_position: Tuple[int, int] = (epos[0][0], epos[1][0])
        ammo: int = int(obs['ammo'])
        blast_strength: int = int(obs['blast_strength'])

        actions = self.legal_moves(my_position, board, bomb_blast_strength[my_position[0], my_position[1]] != 0, ammo)
        non_bomb_actions = actions.copy()
        if Action.Bomb.value in non_bomb_actions:
            non_bomb_actions.remove(Action.Bomb.value)

        action = None

        self.update_enemy_bombs(bomb_life, enemy_position)

        blast_map = util.create_blast_map(board, bomb_blast_strength, bomb_life, bomb_moving_direction, flame_life)
        will_die_preciate = WillDiePredicate(blast_map)

        # check via ASTAR if we can pick up an item
        goal_node = util.astar(board, my_position, non_bomb_actions,
                               FindItemPredicate([Item.Kick.value, Item.ExtraBomb.value, Item.IncrRange.value]),
                               will_die_preciate)
        if goal_node:
            action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
            if path_length >= 12:
                action = None
            else:
                print("Item too far away")
        else:
            print("No collectable item found")

        if action is None:
            tiles_in_range = util.get_in_range(board, my_position, blast_strength)

            if Item.Wood.value in tiles_in_range and Action.Bomb.value in actions:
                new_bomb_life = bomb_life.copy()
                new_bomb_life[my_position] = MAX_BOMB_LIFE
                new_blast_map = util.create_blast_map(board, bomb_blast_strength, new_bomb_life, bomb_moving_direction,
                                                      flame_life)
                new_will_die_predicate = WillDiePredicate(new_blast_map)
                survive_predicate = SurvivePredicate(new_blast_map, MAX_BOMB_LIFE)
                goal_node = util.astar(board, my_position, non_bomb_actions, survive_predicate, new_will_die_predicate)
                if goal_node is not None:
                    action = Action.Bomb.value
                print("Would not survive bomb")
            else:
                print("No wood blastable")

        if action is None:
            # try to approach a wooden tile
            goal_node = util.astar(board, my_position, non_bomb_actions,
                                   FindWoodPredicate(blast_strength, bomb_blast_strength), will_die_preciate)
            if goal_node:
                action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
                if path_length >= 15:
                    action = None
                else:
                    print("Wood too far away")
            else:
                print("No wood approachable")

        print(self.enemy_max_ammo)
        print(self.enemy_bombs)
        print(blast_map)
        time.sleep(1)

        return action if action is not None else Action.Stop

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

    def update_enemy_bombs(self, bomb_life: np.ndarray, enemy_position: Tuple[int, int]):
        self.enemy_bombs = [bomb - 1 for bomb in self.enemy_bombs]
        if len(self.enemy_bombs) > 0 and self.enemy_bombs[-1] == 0:
            self.enemy_ammo += 1
            self.enemy_bombs.pop()

        if bomb_life[enemy_position[0]][enemy_position[1]] == 9:
            if self.enemy_ammo == 0:
                self.enemy_max_ammo += 1
            else:
                self.enemy_ammo -= 1
            self.enemy_bombs.insert(0, 9)

    @staticmethod
    def is_accessible(pos_val: int) -> bool:
        return pos_val in util.ACCESSIBLE_TILES
