from queue import Queue
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Set
import numpy as np

from group02.priority_queue import PriorityQueue
from pommerman import constants
from pommerman.constants import Action, Item

# define tiles we can walk on
ACCESSIBLE_TILES = [Item.Passage.value, Item.Kick.value,
                    Item.IncrRange.value, Item.ExtraBomb.value]

# define tiles that stop a bomb explosion
SOLID_TILES = [Item.Rigid.value, Item.Wood.value]

# define move only actions
MOVE_ACTIONS = [Action.Up.value, Action.Down.value, Action.Left.value, Action.Right.value]

MAX_BOMB_LIFE = 10
MAX_FLAME_TICKS = 2


def next_position(position: Tuple[int, int], action: int) -> Tuple[int, int]:
    """ Returns next position without considering environmental conditions (e.g. rigid tiles)"""
    r, c = position
    if action == constants.Action.Stop.value or action == constants.Action.Bomb.value:
        return r, c
    elif action == constants.Action.Up.value:
        return r - 1, c
    elif action == constants.Action.Down.value:
        return r + 1, c
    elif action == constants.Action.Left.value:
        return r, c - 1
    else:
        return r, c + 1


class Predicate(ABC):
    """ superclass for predicates """

    @abstractmethod
    def test(self, board: np.ndarray, position: Tuple[int, int], cost: int) -> bool:
        raise NotImplementedError()


class FindItemPredicate(Predicate):
    """ predicate is true if item is collected """

    def __init__(self, goal_items: List[int]) -> None:
        self.goal_items = goal_items

    def test(self, board: np.ndarray, position: Tuple[int, int], cost: int) -> bool:
        r, c = position
        return board[r, c] in self.goal_items


class FindWoodPredicate(Predicate):
    """ predicate is true if wooden tile is in blast range """

    def __init__(self, blast_strength: int, bomb_blast_strength: np.ndarray) -> None:
        self.blast_strength = blast_strength
        self.bombs = bomb_blast_strength

    def test(self, board: np.ndarray, position: Tuple[int, int], cost: int) -> bool:
        # check if we can find a wooden tile to blast
        return Item.Wood.value in get_in_range(board, position, self.blast_strength) and \
               self.bombs[position[0], position[1]] == 0.0


class WillDiePredicate(Predicate):
    """ predicate is true if player will die """

    def __init__(self, blast_map: List[List[Set]]) -> None:
        self.blast_map = blast_map

    def test(self, board: np.ndarray, position: Tuple[int, int], cost: int) -> bool:
        # check if we will die in this position at this time
        r, c = position
        return cost in self.blast_map[r][c]


class SurvivePredicate(Predicate):
    """ predicate is true if player survives at cost """

    def __init__(self, blast_map: List[List[Set]], cost: int) -> None:
        self.blast_map = blast_map
        self.cost = cost

    def test(self, board: np.ndarray, position: Tuple[int, int], cost: int) -> bool:
        # check if we will die in this position at this time
        r, c = position
        return len(self.blast_map[r][c]) == 0 or self.cost == cost


class PositionNode:
    """ Position node is only a container """

    def __init__(self, parent: Optional['PositionNode'], position: Tuple[int, int], action: Optional[int]) -> None:
        self.parent = parent
        self.position = position
        self.action = action
        if parent is None:
            self.cost = 0
        else:
            self.cost = parent.cost + 1

    def next(self, action: int) -> Tuple[int, int]:
        return next_position(self.position, action)

    @staticmethod
    def get_path_length_and_action(node: 'PositionNode') -> Tuple[int, int]:
        """ takes a node and returns path length to root node
            and the first action on the path
        """
        if not node:
            raise ValueError("Received None node")
        path_length = 0
        action = 0
        while node.parent:
            path_length += 1
            action = node.action
            node = node.parent
        return action, path_length


def astar(board: np.ndarray, start_position: Tuple[int, int], start_actions: List[int], predicate: Predicate,
          will_die_predicate: WillDiePredicate) -> Optional[PositionNode]:
    """ ASTAR - takes a predicate to find a certain goal node """
    queue = PriorityQueue()
    visited = set()
    start_node = PositionNode(None, start_position, None)
    visited.add(start_position)
    # start actions are actions that have not been pruned
    for action in start_actions:
        next_pos = start_node.next(action)
        visited.add(next_pos)
        node = PositionNode(start_node, next_pos, action)
        if not will_die_predicate.test(board, next_pos, node.cost):
            queue.put(node.cost, node)

    while queue.has_elements():
        node = queue.get()
        if predicate.test(board, node.position, node.cost):
            return node
        for action in [Action.Up.value, Action.Down.value, Action.Left.value, Action.Right.value]:
            next_pos = node.next(action)
            if valid_agent_position(board, next_pos) and next_pos not in visited:
                new_node = PositionNode(node, next_pos, action)
                if not will_die_predicate.test(board, next_pos, node.cost):
                    queue.put(new_node.cost, new_node)
                visited.add(next_pos)
    return None  # no goal node found


def valid_agent_position(board: np.ndarray, pos: Tuple[int, int]) -> bool:
    board_size = len(board)
    r, c = pos
    return 0 <= r < board_size and 0 <= c < board_size and board[r, c] in ACCESSIBLE_TILES


def get_in_range(board: np.ndarray, position: Tuple[int, int], blast_strength: int) -> List[int]:
    """ returns all tiles that are in range of a bomb """
    tiles_in_range = []
    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, blast_strength):
            r = position[0] + row * dist
            c = position[1] + col * dist
            if 0 <= r < len(board) and 0 <= c < len(board):
                tiles_in_range.append(board[r, c])
                if board[r, c] in SOLID_TILES or board[r, c] == Item.Bomb.value:
                    break
            else:
                break
    return tiles_in_range


def get_all_in_range(board: np.ndarray, position: Tuple[int, int], blast_strength: int) -> List[Tuple[int, int]]:
    """ returns all tile positions that are in range of a bomb """
    tiles_in_range = [position]
    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, blast_strength):
            r = position[0] + row * dist
            c = position[1] + col * dist
            if 0 <= r < len(board) and 0 <= c < len(board):
                if board[r, c] in SOLID_TILES or board[r, c] == Item.Bomb.value:
                    break
                tiles_in_range.append((r, c))
            else:
                break
    return tiles_in_range


def create_blast_map(board: np.ndarray, bomb_blast_strength: np.ndarray, bomb_life: np.ndarray,
                     bomb_moving_direction: np.ndarray, flame_life: np.ndarray) -> List[List[Set]]:
    blast_map = [[set() for _ in range(board.shape[1])] for _ in range(board.shape[0])]
    # flames that are currently on the board
    locations = np.where(flame_life > 0)
    for (r, c) in zip(locations[0], locations[1]):
        life = int(flame_life[r, c])
        blast_map[r][c].union(range(life - 1))

    # flames that will be on the board
    locations = np.where(bomb_blast_strength > 0)
    for (r, c) in zip(locations[0], locations[1]):
        position = (r, c)
        life = int(bomb_life[r, c])
        moving_direction = bomb_moving_direction[r, c]
        print("A")
        if moving_direction != Action.Stop.value:
            for _ in range(bomb_life[r, c]):
                new_position = next_position(position, moving_direction)
                if valid_agent_position(board, new_position):
                    position = new_position
                else:
                    break
        tiles = get_all_in_range(board, (r, c), int(bomb_blast_strength[r, c]))
        for (r2, c2) in tiles:
            print("B")
            for i in range(life, life + MAX_FLAME_TICKS):
                blast_map[r2][c2].add(i)
    return blast_map


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
