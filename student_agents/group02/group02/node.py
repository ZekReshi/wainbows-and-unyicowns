import copy
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

ACCESSIBLE_TILES = [Item.Passage.value, Item.Kick.value, Item.IncrRange.value, Item.ExtraBomb.value]


class Node(MCTSNode):
    def __init__(self,
                 board: np.ndarray,
                 own_agent: Agent,
                 opponent_agent: Agent,
                 bombs: List[characters.Bomb],
                 items: Dict[Tuple[int, int], int],
                 flames: List[characters.Flame]) -> None:
        self.total_reward = 0.0
        self.visit_count = 0

        self.board = board
        self.own_agent = own_agent
        self.opponent_agent = opponent_agent
        self.bombs = bombs
        self.items = items
        self.flames = flames

        # here we need to think about pruning (for a particular node)
        # which action combinations do we really want to investigate in our search?
        own_position = self.own_agent.position
        opponent_position = self.opponent_agent.position

        legal_actions = [[a for a in POSSIBLE_ACTIONS if self._is_legal_action(self.own_agent, a)]]
        man_dist = manhattan_dist(own_position, opponent_position)
        legal_actions.insert(self.opponent_agent.aid, [Action.Stop.value] if man_dist > 6 else [a for a in POSSIBLE_ACTIONS if self._is_legal_action(self.opponent_agent, a)])

        self.action_combinations = []
        for a1 in legal_actions[0]:
            for a2 in legal_actions[1]:
                self.action_combinations.append((a1, a2))
                #if a1 != Action.Stop.value:
                    #self.action_combinations.append((a1, a2))
        #print(self.action_combinations)
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
        board, agents, curr_bombs, curr_items, curr_flames = ForwardModel.step(
            actions,
            board,
            agents,
            bombs,
            items,
            flames
        )
        return Node(board, agents[0], agents[1], curr_bombs, curr_items, curr_flames)

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
        return _value_func(root_node, self.board, self.own_agent, self.opponent_agent)


def manhattan_dist(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_wood_on_board_count(board: np.ndarray):
    return np.sum((board == Item.Wood.value))


# if a bomb can kill an agent, give it a high score depending on distance of bomb to agent, otherwise 0
def bomb_kill_score(board: np.ndarray, agent_position, bomb_position: Tuple[int, int], blast_strength: int) -> float:
    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, blast_strength):
            r = bomb_position[0] + row * dist
            c = bomb_position[1] + col * dist
            if 0 <= r < len(board) and 0 <= c < len(board):
                if r == agent_position[0] and c == agent_position[1]:
                    return 0.5 * ((blast_strength - dist) / blast_strength)
                if board[r, c] not in ACCESSIBLE_TILES or board[r, c] == Item.Bomb.value:
                    break
            else:
                break
    return 0


def _value_func(root_node, board, own_agent: Agent, opponent_agent: Agent) -> float:
    # TODO: here you need to assign a value to a game state, for example the evaluation can
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
    attack_mode = get_wood_on_board_count(board) <= 10 or \
                  (not need_bomb_range and not need_ammo and not need_kick)

    # we want to push our agent towards the opponent
    man_dist = manhattan_dist(own_position, opponent_position)
    score += 0.005 * (10 - man_dist) * (2 if attack_mode else 1)  # the closer to the opponent the better

    # we want to collect items (forward model was modified to make this easier)
    score += (own_agent.bomb_range - root_node.own_agent.bomb_range) * (0.2 if need_bomb_range else 0.05)
    score += (own_agent.ammo - root_node.own_agent.ammo) * (0.2 if need_ammo else 0.05)
    score += (own_agent.can_kick - root_node.own_agent.can_kick) * (0.2 if need_kick else 0)

    # since search depth is limited, we need to reward well placed bombs instead
    # of only rewarding collecting items
    woods = {}
    for bomb in own_agent.bombs:
        new_woods = _get_woods_in_range(board, bomb.position, bomb.blast_strength)
        for wood in new_woods:
            woods[wood] = min(woods[wood], bomb.life) if wood in woods else bomb.life
            #print(f"bomb on {bomb.position} with strength {bomb.blast_strength} and life {bomb.life} is gud for wood {wood}")
        #print()
        if attack_mode:
            score += bomb_kill_score(board, opponent_position, bomb.position, bomb.life)
    wood_score = 1.4 ** len(woods)
    for bomb_life in woods.values():
        # default bomb life = 9, rollout depth = 7, 9 - 2 = 2
        wood_score *= 1.1 ** (constants.DEFAULT_BOMB_LIFE - bomb_life + 2)
    score += wood_score * (0.01 if attack_mode else 0.05) - 0.05
    if wood_score * (0.01 if attack_mode else 0.05) - 0.05 > 4:
        print(wood_score * (0.01 if attack_mode else 0.05) - 0.05)
        for bomb in own_agent.bombs:
            print(bomb.position)

    # if in attack_mode, see if we can place game-winning bomb
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
    return [Bomb(bomb.bomber, bomb.position, bomb.life - 1, bomb.blast_strength, bomb.moving_direction) for bomb in bombs]


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
