from queue import Queue
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np

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
    def test(self, board: np.ndarray, position: Tuple[int, int]) -> bool:
        raise NotImplementedError()


class FindItemPredicate(Predicate):
    """ predicate is true if item is collected """

    def __init__(self, goal_items: List[int]) -> None:
        self.goal_items = goal_items

    def test(self, board: np.ndarray, position: Tuple[int, int]) -> bool:
        r, c = position
        return board[r, c] in self.goal_items


class FindWoodPredicate(Predicate):
    """ predicate is true if wooden tile is in blast range """

    def __init__(self, blast_strength: int, bomb_blast_strength: np.ndarray) -> None:
        self.blast_strength = blast_strength
        self.bombs = bomb_blast_strength

    def test(self, board: np.ndarray, position: Tuple[int, int]) -> bool:
        # check if we can find a wooden tile to blast
        return Item.Wood.value in get_in_range(board, position, self.blast_strength) and \
               self.bombs[position[0], position[1]] == 0.0


class PositionNode:
    """ Position node is only a container """

    def __init__(self, parent: Optional['PositionNode'], position: Tuple[int, int], action: Optional[int]) -> None:
        self.parent = parent
        self.position = position
        self.action = action

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


def bfs(board: np.ndarray, start_position: Tuple[int, int], start_actions: List[int], predicate: Predicate) \
        -> Optional[PositionNode]:
    """ BFS - takes a predicate to find a certain goal node """
    queue = Queue()
    visited = set()
    start_node = PositionNode(None, start_position, None)
    visited.add(start_position)
    # start actions are actions that have not been pruned
    for action in start_actions:
        next_pos = start_node.next(action)
        visited.add(next_pos)
        node = PositionNode(start_node, next_pos, action)
        queue.put(node)

    while not queue.empty():
        node = queue.get()
        if predicate.test(board, node.position):
            return node
        for action in [Action.Up.value, Action.Down.value, Action.Left.value, Action.Right.value]:
            next_pos = node.next(action)
            if valid_agent_position(board, next_pos) and next_pos not in visited:
                queue.put(PositionNode(node, next_pos, action))
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


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
