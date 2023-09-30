import numpy as np
from queue import PriorityQueue
import random
from typing import Dict, Any, Tuple, List
from gym import spaces

from pommerman import agents
from pommerman.constants import Item, Action

from . import util
from .util import FindItemPredicate, FindWoodPredicate


class VerySimpleAgent(agents.BaseAgent):
    def __init__(self, *args, **kwargs):
        super(VerySimpleAgent, self).__init__(*args, **kwargs)

    def act(self, obs: Dict[str, Any], action_space: spaces.Discrete) -> int:
        # first we need to reformat the observations space for further processing
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

        # we have extracted all legal moves, we now want to filter those moves,
        # that bring our agent in dangerous situations
        bombs = self._convert_bombs(bomb_life, bomb_blast_strength)
        danger_map = self._get_danger_map(board, bombs, bomb_blast_strength)
        actions = self._prune_dangerous_actions(board, bomb_blast_strength, my_position, actions, danger_map)

        # if <= 1 action is left then we can return here
        if len(actions) == 1:
            return actions[0]
        elif len(actions) == 0:
            # TODO: all moves pruned, maybe you will need some emergency handling here
            return Action.Stop.value

        enemy_dist = util.manhattan_distance(my_position, enemy_position)
        tiles_in_range = util.get_in_range(board, my_position, blast_strength)
        if enemy_dist <= 4:
            # attack heuristic
            # TODO: improve attacking and defending strategy
            # lay bomb if it is not a pruned action and it can hit the enemy
            if enemy.value in tiles_in_range and Action.Bomb.value in actions:
                return Action.Bomb.value
        else:
            # explore and collect heuristic
            # TODO: you might find more efficient and accurate algorithms here
            # check via BFS if we can pick up an item
            goal_node = util.bfs(board, my_position, actions,
                                 FindItemPredicate([Item.Kick.value, Item.ExtraBomb.value, Item.IncrRange.value]))
            if goal_node:
                action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
                if path_length < 12:
                    return action

            if Item.Wood.value in tiles_in_range and Action.Bomb.value in actions:
                return Action.Bomb.value

            # try to approach a wooden tile
            goal_node = util.bfs(board, my_position, actions,
                                 FindWoodPredicate(blast_strength, bomb_blast_strength))
            if goal_node:
                action, path_length = util.PositionNode.get_path_length_and_action(goal_node)
                if path_length < 15:
                    return action

        # final strategy if all others fail - randomly choosing move
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

    @staticmethod
    def _convert_bombs(bomb_life: np.ndarray, bomb_blast_strength: np.ndarray) -> PriorityQueue:
        """Convert bomb matrices in bomb queue sorted by bomb life
            sorting bombs makes calculating the danger map faster afterwards
        """
        bombs = PriorityQueue()
        locations = np.where(bomb_blast_strength > 0)
        for r, c in zip(locations[0], locations[1]):
            bombs.put((int(bomb_life[r, c]), int(bomb_blast_strength[r, c]), (r, c)))
        return bombs

    @staticmethod
    def _prune_dangerous_actions(board: np.ndarray, bomb_blast_strength: np.ndarray, position: Tuple[int, int],
                                 actions: List[int], danger_map: np.ndarray) -> List[int]:
        # TODO: finding good pruning rules here is extremely important,
        #  the version currently used is only a very rough approximation
        pruned_actions = []
        for action in actions:
            # we already know that r and c are in bounds here
            r, c = util.next_position(position, action)
            # the following pruning rules are only heuristics (not exact rules)
            # always move away from a bomb
            if action == Action.Stop.value and bomb_blast_strength[r, c] != 0.0:
                continue
            if danger_map[r, c] <= 2:
                # too dangerous if bomb explosion happens in 2 steps
                continue
            elif action == Action.Bomb.value:
                # it is too dangerous to trigger chain reactions - avoid them
                if danger_map[r, c] < util.MAX_BOMB_LIFE:
                    continue

            # check if agent is locked in
            down_cond = r + 1 >= len(board) or \
                board[r + 1, c] in util.SOLID_TILES or \
                bomb_blast_strength[r + 1, c] != 0
            up_cond = r - 1 < 0 or \
                board[r - 1, c] in util.SOLID_TILES or \
                bomb_blast_strength[r - 1, c] != 0
            right_cond = c + 1 >= len(board) or \
                board[r, c + 1] in util.SOLID_TILES or \
                bomb_blast_strength[r, c + 1] != 0
            left_cond = c - 1 < 0 or \
                board[r, c - 1] in util.SOLID_TILES or \
                bomb_blast_strength[r, c - 1] != 0

            if not(down_cond and up_cond and right_cond and left_cond):
                pruned_actions.append(action)
        return pruned_actions

    @staticmethod
    def _get_danger_map(board: np.ndarray, bombs: PriorityQueue, bomb_blast_strength: np.ndarray) -> np.ndarray:
        """Returns a map that shows next bomb explosion for all fields"""
        board_size = len(board)
        danger_map = np.zeros(shape=(board_size, board_size), dtype=np.int) + util.MAX_BOMB_LIFE
        while not bombs.empty():
            bomb = bombs.get()  # get bomb with lowest bomb life in queue
            # unpack tuple values
            bomb_life = bomb[0]
            bomb_range = bomb[1]
            bomb_pos = bomb[2]
            if danger_map[bomb_pos[0], bomb_pos[1]] < bomb_life:
                # bomb already triggered by other bomb with shorter bomb_life, continue with next bomb in queue
                continue
            else:
                danger_map[bomb_pos[0], bomb_pos[1]] = bomb_life
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in range(1, bomb_range):
                    r = bomb_pos[0] + row * dist
                    c = bomb_pos[1] + col * dist
                    if r < 0 or r >= board_size or c < 0 or c >= board_size:
                        # out of border
                        break
                    if bomb_blast_strength[r, c] != 0:
                        # we hit another bomb
                        if bomb_life < danger_map[r, c]:
                            bombs.put((bomb_life, int(bomb_blast_strength[r, c]), (r, c)))
                        break
                    elif board[r, c] in util.SOLID_TILES:
                        # solid tile stops bomb
                        danger_map[r, c] = min(danger_map[r, c], bomb_life)
                        break
                    else:
                        # update danger map
                        danger_map[r, c] = min(danger_map[r, c], bomb_life)
        return danger_map
