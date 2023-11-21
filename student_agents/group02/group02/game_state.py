import numpy as np
from typing import Dict, Any, Tuple, List

from pommerman import constants, utility
from pommerman import characters
from pommerman.characters import Bomb
from pommerman.constants import Item


class Agent:
    def __init__(self, aid: int, board_id: int, position: Tuple[int, int], can_kick: bool, max_ammo: int, ammo: int, bomb_range: int,
                 alive: bool):
        self.aid = aid
        self.board_id = board_id
        self.position = position
        self.can_kick = can_kick
        self.max_ammo = max_ammo
        self.ammo = ammo
        self.bomb_range = bomb_range
        self.alive = alive

    def __str__(self):
        return f"ID: {self.aid}, Board ID: {self.board_id}, Position: {self.position}, Kick: {self.can_kick}, Max ammo: {self.max_ammo}, Ammo: {self.ammo}, Bomb Range: {self.bomb_range}, Alive: {self.alive}"

    def get_next_position(self, direction) -> Tuple[int, int]:
        action = constants.Action(direction)
        return utility.get_next_position(self.position, action)

    def move(self, direction):
        self.position = self.get_next_position(direction)

    def maybe_lay_bomb(self):
        if self.ammo > 0:
            self.ammo -= 1
            return Bomb(self, self.position, constants.DEFAULT_BOMB_LIFE + 1,
                        self.bomb_range)
        return None

    def die(self):
        self.alive = False

    def incr_ammo(self):
        self.ammo = min(self.ammo + 1, 10)

    def pick_up(self, item, max_blast_strength):
        if item == constants.Item.ExtraBomb:
            self.incr_ammo()
        elif item == constants.Item.IncrRange:
            self.bomb_range = min(self.bomb_range + 1,
                                  max_blast_strength)
        elif item == constants.Item.Kick:
            self.can_kick = True


# we have to create a game state from the observations
# in this example we only use an approximation of the game state, this can be improved
# approximations are:
#  - flames: initialized with 2 ticks (might be 1)
#  - agents: initialized with ammo=2 (might be more or less)
#  - bombs: we do not consider, which agent placed the bomb,
#           after explosion this would increase the agent's ammo again
#  - items: we do not know if an item is placed below a removable tile
def game_state_from_obs(
        obs: Dict[str, Any],
        me: Agent, opponent: Agent,
        prev_board: np.ndarray = None) \
        -> Tuple[np.ndarray,
        Agent,
        Agent,
        List[characters.Bomb],
        Dict[Tuple[int, int], int],
        List[characters.Flame]]:
    # TODO: think about removing some of the approximations and replacing them
    #   with exact values (e.g. tracking own and opponent's ammo and using exact flame life)
    board = obs["board"]
    #print(obs["bomb_life"].max())
    return (board,
            convert_me(board, obs["ammo"], me),
            convert_opponent(obs["board"], prev_board, opponent),
            convert_bombs(np.array(obs["bomb_blast_strength"]), np.array(obs["bomb_life"])),
            convert_items(board),
            convert_flames(board, obs["flame_life"]))


def convert_bombs(strength_map: np.ndarray, life_map: np.ndarray) -> List[characters.Bomb]:
    """ converts bomb matrices into bomb object list that can be fed to the forward model """
    ret = []
    locations = np.where(strength_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append(
            {'position': (r, c), 'blast_strength': int(strength_map[(r, c)]), 'bomb_life': int(life_map[(r, c)]),
             'moving_direction': None})
    return make_bomb_items(ret)


def make_bomb_items(ret: List[Dict[str, Any]]) -> List[characters.Bomb]:
    bomb_obj_list = []

    for i in ret:
        bomber = characters.Bomber()  # dummy bomber is used here instead of the actual agent
        bomb_obj_list.append(
            characters.Bomb(bomber, i['position'], i['bomb_life'], i['blast_strength'],
                            i['moving_direction']))
    return bomb_obj_list


def convert_me(board: np.ndarray, ammo: int, me: Agent) -> Agent:
    locations = np.where(board == me.board_id)
    if len(locations) == 0:
        me.alive = False
    me.position = (locations[0][0], locations[1][0])
    me.ammo = ammo
    return me


def convert_opponent(board: np.ndarray, prev_board: np.ndarray, opponent: Agent) -> Agent:
    locations = np.where(board == opponent.board_id)
    if len(locations) == 0:
        opponent.alive = False
    position = (locations[0][0], locations[1][0])
    if board[position] == Item.Bomb.value:
        opponent.ammo -= 1
    if prev_board is not None:
        if prev_board[position] == Item.Kick.value:
            opponent.can_kick = True
        elif prev_board[position] == Item.ExtraBomb.value:
            opponent.max_ammo += 1
            opponent.ammo += 1
        elif prev_board[position] == Item.IncrRange.value:
            opponent.bomb_range += 1
    opponent.position = position
    return opponent


def convert_items(board: np.ndarray) -> Dict[Tuple[int, int], int]:
    """ converts all visible items to a dictionary """
    ret = {}
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            v = board[r][c]
            if v in [constants.Item.ExtraBomb.value,
                     constants.Item.IncrRange.value,
                     constants.Item.Kick.value]:
                ret[(r, c)] = v
    return ret


def convert_flames(board: np.ndarray, flame_life: np.ndarray) -> List[characters.Flame]:
    """ creates a list of flame objects - initialized with flame_life=2 """
    ret = []
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append(characters.Flame((r, c), int(flame_life[(r, c)])))
    return ret
