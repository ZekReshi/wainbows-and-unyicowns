import copy
import random
import numpy as np
from typing import Dict, Tuple, List, Optional

from pommerman.constants import Action
from pommerman.agents import DummyAgent
from pommerman.forward_model import ForwardModel
from pommerman import constants, characters, agents
from pommerman.constants import Item, POSSIBLE_ACTIONS

from .mcts import MCTSNode

ACCESSIBLE_TILES = [Item.Passage.value, Item.Kick.value, Item.IncrRange.value, Item.ExtraBomb.value]


class Node(MCTSNode):
    def __init__(self,
                 state: Tuple[
                            np.ndarray,
                            List[agents.DummyAgent],
                            List[characters.Bomb],
                            Dict[Tuple[int, int], int],
                            List[characters.Flame]
                        ],
                 agent_id: int) -> None:
        self.total_reward = 0.0
        self.visit_count = 0
        # state is a list of: 0. Board, 1. Agents, 2. Bombs, 3. Items, 4. Flames
        self.state = state
        self.agent_id = agent_id

        # here we need to think about pruning (for a particular node)
        # which action combinations do we really want to investigate in our search?
        self.own_legal_actions = []
        self.own_illegal_actions = []
        self.enemy_legal_actions = []
        self.enemy_illegal_actions = []
        self.action_combinations: List[Tuple[int, int]] = \
            [(a1, a2) for a1 in POSSIBLE_ACTIONS for a2 in POSSIBLE_ACTIONS if not self.prune((a1, a2))]
        # dictionary to store children according to actions performed
        self.children: Dict[Tuple[int, int], 'Node'] = dict()

    def prune(self, actions: Tuple[int, int]) -> bool:
        # TODO: here you can think about more complex stategies to prune moves,
        #   which allows you to create deeper search trees (very important!)
        # remember: two agents -> ids: 0 and 1
        own_agent = self.state[1][self.agent_id]
        opponent_agent = self.state[1][1 - self.agent_id]

        # if an action has been decided to be illegal once, it will obviously be illegal again
        if actions[self.agent_id] in self.own_illegal_actions or actions[opponent_agent.agent_id]:
            return False

        own_position = own_agent.position
        opponent_position = opponent_agent.position
        man_dist = manhattan_dist(own_position, opponent_position)
        if man_dist > 6 and actions[opponent_agent.agent_id] != Action.Stop.value:
            # we do not model the opponent, if it is more than 6 steps away
            return True

        # a lot of moves (e.g. bumping into a wall or wooden tile) actually result in stop moves
        # we do not have to consider, since they lead to the same result as actually playing a stop move
        if actions[self.agent_id] not in self.own_legal_actions:
            if not self._is_legal_action(own_position, actions[self.agent_id]):
                self.own_illegal_actions.append(actions[self.agent_id])
                return False
            self.own_legal_actions.append(actions[self.agent_id])
        if actions[opponent_agent.agent_id] not in self.enemy_legal_actions:
            if not self._is_legal_action(opponent_position, actions[opponent_agent.agent_id]):
                self.enemy_illegal_actions.append(actions[opponent_agent.agent_id])
                return False
            self.enemy_legal_actions.append(actions[opponent_agent.agent_id])
        return True

    def _is_legal_action(self, position: Tuple[int, int], action: int) -> bool:
        """ prune moves that lead to stop move"""
        if action == Action.Stop.value:
            return True
        board = self.state[0]
        bombs = self.state[2]
        bombs = [bomb.position for bomb in bombs]
        row = position[0]
        col = position[1]
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
        if row < 0 or row >= len(board) or col < 0 or col >= len(board):
            return False

        # check if action leads agent to accessible tile
        if board[row, col] in [Item.Wood.value, Item.Rigid.value]:
            return False

        return True

    def _forward(self, actions: Tuple[int, int]) -> 'Node':
        """ applies the actions to obtain the next game state """
        # since the forward model directly modifies the parameters, we have to provide copies
        board = copy.deepcopy(self.state[0])
        agents = _copy_agents(self.state[1])
        bombs = _copy_bombs(self.state[2])
        items = copy.deepcopy(self.state[3])
        flames = _copy_flames(self.state[4])
        board, curr_agents, curr_bombs, curr_items, curr_flames = ForwardModel.step(
            actions,
            board,
            agents,
            bombs,
            items,
            flames
        )
        return Node((board, curr_agents, curr_bombs, curr_items, curr_flames), self.agent_id)

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
        alive = [agent for agent in self.state[1] if agent.is_alive]
        return len(alive) != 2

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

    def reward(self, root_state) -> float:
        # we do not want to role out games until the end,
        # since pommerman games can last for 800 steps, therefore we need to define a value function,
        # which assigns a numeric value to state (how "desirable" is the state?)
        return _value_func(self.state, root_state, self.agent_id)


def manhattan_dist(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def _value_func(state, root_state, agent_id) -> float:
    # TODO: here you need to assign a value to a game state, for example the evaluation can
    #   be based on the number of blasted clouds, the number of collected items, the distance to the opponent, ...
    # state is a list of: 0. Board, 1. Agents, 2. Bombs, 3. Items, 4. Flames
    # an example how a numerical value can be derived:
    board = state[0]
    agents = state[1]
    own_agent = agents[agent_id]
    opponent_agent = agents[1-agent_id]
    root_own_agent = root_state[1][agent_id]
    assert own_agent, root_own_agent
    # check if own agent is dead
    if not own_agent.is_alive:
        return -1.0
    # check if opponent has been destroyed
    elif not opponent_agent.is_alive:
        return 1.0

    score = 0.0  # game is not over yet, we have to think about additional evaluation criteria

    own_position = own_agent.position
    opponent_position = opponent_agent.position

    # if agent cannot move in any direction than it's locked up either by a bomb,
    # or the opponent agent -> very bad position
    down_cond = own_position[0] + 1 >= len(board) or \
        board[own_position[0] + 1][own_position[1]] not in ACCESSIBLE_TILES
    up_cond = own_position[0] - 1 < 0 or \
        board[own_position[0] - 1][own_position[1]] not in ACCESSIBLE_TILES
    right_cond = own_position[1] + 1 >= len(board) or \
        board[own_position[0]][own_position[1] + 1] not in ACCESSIBLE_TILES
    left_cond = own_position[1] - 1 < 0 or \
        board[own_position[0]][own_position[1] - 1] not in ACCESSIBLE_TILES

    if down_cond and up_cond and right_cond and left_cond:
        score += -0.5

    # we want to push our agent towards the opponent
    man_dist = manhattan_dist(own_position, opponent_position)
    score += 0.005*(10-man_dist)  # the closer to the opponent the better

    # we want to collect items (forward model was modified to make this easier)
    score += own_agent.picked_up_items * 0.05

    # since search depth is limited, we need to reward well placed bombs instead
    # of only rewarding collecting items
    for bomb in state[2]:
        tiles = _get_in_range(board, bomb.position, bomb.blast_strength)
        for tile in tiles:
            if tile == Item.Wood.value:
                score += 0.05
    return score


def _get_in_range(board: np.ndarray, position: Tuple[int, int], blast_strength: int) -> List[int]:
    """ returns all tiles that are in range of a bomb """
    tiles_in_range = []
    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, blast_strength):
            r = position[0] + row * dist
            c = position[1] + col * dist
            if 0 <= r < len(board) and 0 <= c < len(board):
                tiles_in_range.append(board[r, c])
                if board[r, c] not in ACCESSIBLE_TILES or board[r, c] == Item.Bomb.value:
                    break
            else:
                break
    return tiles_in_range


def _copy_agents(agents_to_copy: List[agents.DummyAgent]) -> List[agents.DummyAgent]:
    """ copy agents of the current node """
    agents_copy = []
    for agent in agents_to_copy:
        agt = DummyAgent()
        agt.init_agent(agent.agent_id, constants.GameType.FFA)
        agt.set_start_position(agent.position)
        agt.reset(
            ammo=agent.ammo,
            is_alive=agent.is_alive,
            blast_strength=agent.blast_strength,
            can_kick=agent.can_kick
        )
        agt.picked_up_items = agent.picked_up_items
        agents_copy.append(agt)
    return agents_copy


def _copy_bombs(bombs: List[characters.Bomb]) -> List[characters.Bomb]:
    """ copy bombs of the current node """
    bombs_copy = []
    for bomb in bombs:
        bomber = characters.Bomber()
        bombs_copy.append(
            characters.Bomb(bomber, bomb.position, bomb.life, bomb.blast_strength,
                            bomb.moving_direction)
        )

    return bombs_copy


def _copy_flames(flames: List[characters.Flame]) -> List[characters.Flame]:
    """ copy flames of the current node """
    flames_copy = []
    for flame in flames:
        flames_copy.append(
            characters.Flame(flame.position, flame.life)
        )
    return flames_copy
