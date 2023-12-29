import math
import random
import numpy as np
from typing import Dict, Tuple, List, Optional


from pommerman.characters import Bomb, Bomber, Flame
from pommerman.constants import Action
from pommerman import characters, constants
from pommerman.constants import Item, POSSIBLE_ACTIONS
from .game_state import Agent

from .mcts import MCTSNode
from .my_forward_model import ForwardModel
from .priority_queue import PriorityQueue

ACCESSIBLE_TILES = [Item.Passage.value, Item.Kick.value, Item.IncrRange.value, Item.ExtraBomb.value]




class Node(MCTSNode):
    def __init__(self,
                 board: np.ndarray,
                 own_agent: Agent,
                 opponent_agent: Agent,
                 bombs: List[characters.Bomb],
                 items: Dict[Tuple[int, int], int],
                 flames: List[characters.Flame],
                 is_deterministic: bool) -> None:
        self.total_reward = 0.0
        self.visit_count = 0

        self.board = board
        self.own_agent = own_agent
        self.opponent_agent = opponent_agent
        self.bombs = bombs
        self.items = items
        self.flames = flames
        self.is_deterministic = is_deterministic
        self.a_star_cache:Dict[Tuple[int, int], Dict[Tuple[int, int], List[Tuple[int, int]]]] = dict()
        # here we need to think about pruning (for a particular node)
        # which action combinations do we really want to investigate in our search?
        own_position = self.own_agent.position
        opponent_position = self.opponent_agent.position

        legal_actions = [[a for a in POSSIBLE_ACTIONS if self._is_legal_action(self.own_agent, a)]]
        man_dist = self.manhattan_dist(own_position, opponent_position)
        legal_actions.insert(self.opponent_agent.aid,
                             [Action.Stop.value] if man_dist > 6 else [a for a in POSSIBLE_ACTIONS if
                                                                       self._is_legal_action(self.opponent_agent, a)])

        self.action_combinations = []
        for a1 in legal_actions[0]:
            for a2 in legal_actions[1]:
                self.action_combinations.append((a1, a2))
                # if a1 != Action.Stop.value:
                # self.action_combinations.append((a1, a2))
        # print(self.action_combinations)
        # dictionary to store children according to actions performed
        self.children: Dict[Tuple[int, int], 'Node'] = dict()

    def _is_legal_action(self, agent: Agent, action: int) -> bool:
        """ prune moves that lead to stop move"""
        if action == Action.Stop.value:
            return True

        bombs = [bomb.position for bomb in agent.bombs]
        row = agent.position[0]
        col = agent.position[1]
        # if it a bomb move, check if there is already a bomb planted on this field
        if action == Action.Bomb.value and (row, col) in bombs:
            return False

        if action == Action.Up.value:
            row -= 1
        elif action == Action.Down.value:
            row += 1
        elif action == Action.Left.value:
            col -= 1
        elif action == Action.Right.value:
            col += 1

        # check if action leads to out of bounds position
        if row < 0 or row >= len(self.board) or col < 0 or col >= len(self.board):
            return False

        # check if action leads agent to accessible tile
        if self.board[row, col] in [Item.Wood.value, Item.Rigid.value]:
            return False

        return True

    def _forward(self, actions: Tuple[int, int]) -> 'Node':
        """ applies the actions to obtain the next game state """
        # since the forward model directly modifies the parameters, we have to provide copies
        board = np.copy(self.board)
        agents = _copy_agents([self.own_agent, self.opponent_agent])
        bombs = _copy_bombs(self.bombs)
        items = self.items.copy()
        flames = _copy_flames(self.flames)
        board, agents, curr_bombs, curr_items, curr_flames, is_deterministic = ForwardModel.step(
            actions,
            board,
            agents,
            bombs,
            items,
            flames
        )
        return Node(board, agents[0], agents[1], curr_bombs, curr_items, curr_flames, is_deterministic)

    def step(self, actions: Tuple[int, int]):
        matching_child = None
        for action_pair, child in list(self.children.items()):
            if not child.is_deterministic or action_pair != actions:
                del self.children[action_pair]
                del child
                continue
            matching_child = child
        return matching_child

    def find_random_child(self) -> 'Node':
        """ returns a random child, expands the child if it has not already been done """
        actions = random.choice(self.action_combinations)
        if actions in self.children.keys():
            return self.children[actions]
        else:
            child = self._forward(actions)
            self.children[actions] = child  # add child to expanded nodes
            return child

    def get_children(self) -> Dict[Tuple[int, int], 'Node']:
        return self.children

    def get_unexplored(self) -> Optional['Node']:
        """ returns a randomly chosen unexplored action pair, or None """
        unexplored_actions = [actions for actions in self.action_combinations if actions not in self.children.keys()]
        if not unexplored_actions:
            return None
        actions = random.choice(unexplored_actions)
        child = self._forward(actions)
        self.children[actions] = child
        return child

    def is_terminal(self):
        return not (self.own_agent.alive and self.opponent_agent.alive)

    def get_total_reward(self) -> float:
        """ Returns Total reward of node (Q) """
        return self.total_reward

    def incr_reward(self, reward: float) -> None:
        """ Update reward of node in backpropagation step of MCTS """
        self.total_reward += reward

    def get_visit_count(self) -> int:
        """ Returns Total number of times visited this node (N) """
        return self.visit_count

    def incr_visit_count(self) -> None:
        self.visit_count += 1

    def reward(self, root_node) -> float:
        # we do not want to role out games until the end,
        # since pommerman games can last for 800 steps, therefore we need to define a value function,
        # which assigns a numeric value to state (how "desirable" is the state?)
        return self._value_func(root_node, self.board, self.own_agent, self.opponent_agent)


    def manhattan_dist(self,pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    def a_star_path(self,pos1: Tuple[int, int], pos2: Tuple[int, int], board: np.ndarray) -> List[Tuple[int, int]]:
        if pos1 in self.a_star_cache and pos2 in self.a_star_cache[pos1]:
            return self.a_star_cache[pos1][pos2]
        goal = pos2
        current = pos1
        fringe = PriorityQueue()
        visited = set()
        came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        while not current == goal:
            visited.add(current)

            successors = [
                (current[0] - 1, current[1]),
                (current[0] + 1, current[1]),
                (current[0], current[1] - 1),
                (current[0], current[1] + 1)

            ]

            for node in successors:
                if node[0] < 0 or node[1] < 0:
                    continue
                if node[0] >= board.shape[0] or node[1] >= board.shape[1]:
                    continue
                if board[node[0]][node[1]] in [Item.Rigid.value]:
                    continue
                if node not in visited and node not in fringe:
                    fringe.put(heuristic(node, goal) + 1, node)
            old = current
            if not fringe.has_elements():
                return []
            current = fringe.get()
            if current != pos1:
                came_from[current] = old
        path = []
        current = pos2
        while current != pos1:
            path.append(current)
            current = came_from[current]
        path.reverse()
        if pos1 not in self.a_star_cache:
            self.a_star_cache[pos1]= dict()
        self.a_star_cache[pos1][pos2] = path
        return path


    def astar_dist(self,pos1: Tuple[int, int], pos2: Tuple[int, int], board) -> int:
        length = 0
        path = self.a_star_path(pos1,pos2,board)
        for node in path:
            length+=1
            if board[node[0]][node[1]] in [Item.Flames.value,Item.Bomb.value]:
                break
        return length



    def _value_func(self,root_node, board, own_agent: Agent, opponent_agent: Agent) -> float:
        # TODO: here you need to assign a value seto a game state, for example the evaluation can
        #   be based on the number of blasted clouds, the number of collected items, the distance to the opponent, ...
        # state is a list of: 0. Board, 1. Agents, 2. Bombs, 3. Items, 4. Flames
        # an example how a numerical value can be derived:
        # check if own agent is dead
        if not own_agent.alive:
            return -1.0
        # check if opponent has been destroyed
        elif not opponent_agent.alive:
            return 1.0

        score = 0.0  # game is not over yet, we have to think about additional evaluation criteria

        own_position = own_agent.position
        opponent_position = opponent_agent.position

        # if agent cannot move in any direction than it's locked up either by a bomb,
        # or the opponent agent -> very bad position
        cant_move = (own_position[0] + 1 >= len(board) or
                     board[own_position[0] + 1][own_position[1]] not in ACCESSIBLE_TILES) and \
                    (own_position[0] - 1 < 0 or
                     board[own_position[0] - 1][own_position[1]] not in ACCESSIBLE_TILES) and \
                    (own_position[1] + 1 >= len(board) or
                     board[own_position[0]][own_position[1] + 1] not in ACCESSIBLE_TILES) and \
                    (own_position[1] - 1 < 0 or
                     board[own_position[0]][own_position[1] - 1] not in ACCESSIBLE_TILES)

        if cant_move:
            if not own_agent.can_kick:
                if own_position[0] + 1 < len(board) and \
                        board[own_position[0] + 1][own_position[1]] == Item.Bomb.value or \
                        own_position[0] - 1 >= 0 and \
                        board[own_position[0] - 1][own_position[1]] == Item.Bomb.value or \
                        own_position[1] + 1 < len(board) and \
                        board[own_position[0]][own_position[1] + 1] == Item.Bomb.value or \
                        own_position[1] - 1 >= 0 and \
                        board[own_position[0]][own_position[1] - 1] == Item.Bomb.value:
                    return -1

            score -= 0.5

        # if there are not many woods left to destroy or we have all upgrades focus on getting to the enemy and place bomb
        need_bomb_range = own_agent.bomb_range < 8
        need_ammo = own_agent.ammo < 4
        need_kick = not own_agent.can_kick
        attack_mode = get_wood_on_board_count(board) <= 12 or \
                      (not need_bomb_range and not need_ammo and not need_kick)

        MOVE_ENEMY_WEIGHT = 0.05 if not attack_mode else 0.7
        COLLECT_ITEM_WEIGHT =0.6 if not attack_mode else 0.05
        BOMB_WEIGHT= 0.35 if not attack_mode else 0.25
        # we want to push our agent towards the opponent
        move_enemy_score = 0
        dist = self.astar_dist(own_position, opponent_position, board)
        if dist >1:
            move_enemy_score = max(10 - ((dist-2)//2),0)

        # we want to collect items (forward model was modified to make this easier)
        collect_item_score = get_collect_item_score(own_agent, root_node.own_agent)

        # since search depth is limited, we need to reward well placed bombs instead
        # of only rewarding collecting items
        #b
        bomb_scores = []

        for bomb in own_agent.bombs:
            new_woods = _get_woods_in_range(board, bomb.position, bomb.blast_strength)
            # if in attack_mode, see if we can place game-winning bomb
            bomb_score = 0
            if attack_mode:
                bomb_score += bomb_kill_score(board, opponent_position, bomb.position, bomb.blast_strength)
            else:
                bomb_score += (min(10,len(new_woods)+8) if len(new_woods)>0 else 0) * min(1,((constants.DEFAULT_BOMB_LIFE - bomb.life+2)/constants.DEFAULT_BOMB_LIFE))
            bomb_scores.append(bomb_score)
        #for bomb_life in woods.values():
        #    # default bomb life = 9, rollout depth = 7, 9 - 2 = 2
        #    score +=  (0.1 if attack_mode else 0.5) * ((constants.DEFAULT_BOMB_LIFE - bomb_life + 2)/constants.DEFAULT_BOMB_LIFE)

        score += MOVE_ENEMY_WEIGHT*move_enemy_score/10
        score += COLLECT_ITEM_WEIGHT*collect_item_score/10
        if len(bomb_scores) > 0:
            score += BOMB_WEIGHT * (max(bomb_scores) if attack_mode else np.average(bomb_scores)) / 10


        return score


def _get_woods_in_range(board: np.ndarray, position: Tuple[int, int], blast_strength: int) -> List[Tuple[int, int]]:
    """ returns all tiles that are in range of a bomb """
    woods_in_range = []
    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, blast_strength):
            r = position[0] + row * dist
            c = position[1] + col * dist
            if 0 <= r < len(board) and 0 <= c < len(board):
                if board[r, c] == Item.Wood.value:
                    woods_in_range.append((r, c))
                    break
                if board[r, c] not in ACCESSIBLE_TILES or board[r, c] == Item.Bomb.value:
                    break
            else:
                break
    return woods_in_range


def _copy_agents(agents_to_copy: List[Agent]) -> List[Agent]:
    """ copy agents of the current node """
    agents_copy = []
    for agent in agents_to_copy:
        agt = Agent(
            agent.aid,
            agent.board_id,
            agent.position,
            agent.can_kick,
            agent.ammo,
            _copy_agent_bombs(agent.bombs),
            agent.bomb_range,
            agent.alive
        )
        agents_copy.append(agt)
    return agents_copy


def _copy_agent_bombs(bombs: List[Bomb]) -> List[Bomb]:
    return [Bomb(bomb.bomber, bomb.position, bomb.life - 1, bomb.blast_strength, bomb.moving_direction) for bomb in
            bombs]


def _copy_bombs(bombs: List[Bomb]) -> List[Bomb]:
    """ copy bombs of the current node """
    return [Bomb(Bomber(), bomb.position, bomb.life, bomb.blast_strength, bomb.moving_direction) for bomb in bombs]


def _copy_flames(flames: List[Flame]) -> List[Flame]:
    """ copy flames of the current node """
    flames_copy = []
    for flame in flames:
        flames_copy.append(
            characters.Flame(flame.position, flame.life)
        )
    return flames_copy

def heuristic(current, goal):
    cy, cx = current
    gy, gx = goal
    return math.sqrt((cy - gy) ** 2 + (cx - gx) ** 2)

def get_wood_on_board_count(board: np.ndarray):
    return np.sum((board == Item.Wood.value))

# if a bomb can kill an agent, give it a high score depending on distance of bomb to agent, otherwise 0
def bomb_kill_score(board: np.ndarray, agent_position, bomb_position: Tuple[int, int],
                    blast_strength: int) -> float:
    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, min(5,blast_strength)):
            r = bomb_position[0] + row * dist
            c = bomb_position[1] + col * dist
            if 0 <= r < len(board) and 0 <= c < len(board):
                if r == agent_position[0] and c == agent_position[1]:
                    return min(10 - (dist-2),10)
                if board[r, c] not in ACCESSIBLE_TILES or board[r, c] == Item.Bomb.value:
                    break
            else:
                break
    return 0

def get_collect_item_score(own_agent, own_agent1):
    range_diff = (own_agent.bomb_range - own_agent1.bomb_range)
    ammo_diff =  (own_agent.ammo - own_agent1.ammo)
    kick_diff = (own_agent.can_kick - own_agent1.can_kick)
    diff_sum  = range_diff + ammo_diff + kick_diff
    if diff_sum > 0:
        return 10
    else:
        return 0

