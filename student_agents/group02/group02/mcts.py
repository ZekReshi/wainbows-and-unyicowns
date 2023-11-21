"""
A nice practical MCTS explanation:
   https://www.youtube.com/watch?v=UXW2yZndl7U
This implementation is based on:
   https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
import math
from gym import spaces
import numpy as np
from typing import Tuple, List, Dict, Optional

from group02.game_state import Agent
from pommerman import characters, agents


class MCTS:
    # Monte Carlo tree searcher. First rollout the tree then choose a move.

    # TODO: you can experiment with the values rollout_depth (depth of simulations)
    #  and exploration_weight here, they are not tuned for Pommerman
    def __init__(self,
                 action_space: spaces.Discrete,
                 agent_id: int,
                 root_state: Tuple[
                            np.ndarray,
                            Agent,
                            Agent,
                            List[characters.Bomb],
                            Dict[Tuple[int, int], int],
                            List[characters.Flame]
                        ],
                 rollout_depth: int = 4,
                 exploration_weight: float = 1) -> None:
        self.action_space = action_space
        self.root_state = root_state
        self.agent_id = agent_id
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight  # used in uct formula

    def choose(self, node: 'MCTSNode') -> int:
        """ Choose the best successor of node. (Choose an action) """
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        children: Dict[Tuple[int, int], 'MCTSNode'] = node.get_children()
        if len(children) == 0:
            # choose a move randomly, should hopefully never happen
            return self.action_space.sample()

        def score(key: Tuple[int, int]):
            n = children[key]
            if n.get_visit_count() == 0:
                return float("-inf")  # avoid unseen moves, should also never happen
            return n.get_total_reward() / n.get_visit_count()  # average reward

        return max(children.keys(), key=score)[self.agent_id]

    def do_rollout(self, node: 'MCTSNode') -> None:
        """ Execute one tree update step: select, expand, simulate, backpropagate """
        path: List['MCTSNode'] = self._select_and_expand(node)
        leaf: 'MCTSNode' = path[-1]
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select_and_expand(self, node: 'MCTSNode') -> List['MCTSNode']:
        """ Find an unexplored descendent of node """
        path = []
        while True:
            path.append(node)
            # leaf node?
            if node.is_terminal():
                return path
            # if there are unexplored child nodes left, pick one of them at random,
            # because they have highest uct value
            unexplored = node.get_unexplored()  # expansion takes place here
            if unexplored is not None:
                path.append(unexplored)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _simulate(self, node: 'MCTSNode') -> float:
        """ performs simulation and returns reward from value function """
        depth = 0
        while not node.is_terminal() and depth < self.rollout_depth:
            node = node.find_random_child()  # default policy is random policy
            depth += 1
        return node.reward()

    def _backpropagate(self, path: List['MCTSNode'], reward: float) -> None:
        # Send the reward back up to the ancestors of the leaf
        for node in reversed(path):
            node.incr_visit_count()
            node.incr_reward(reward)

    def _uct_select(self, node: 'MCTSNode') -> 'MCTSNode':
        """ Select a child of node, balancing exploration & exploitation """

        children = node.get_children().values()
        visit_count = node.get_visit_count()
        log_n_vertex = math.log(visit_count)

        def uct(n: 'MCTSNode'):
            q = n.get_total_reward()
            ni = n.get_visit_count()
            if ni == 0:
                return float('inf')
            "Upper confidence bound for trees"
            return q / ni + self.exploration_weight * math.sqrt(
                log_n_vertex / ni
            )

        return max(children, key=uct)


# some utility functions to debug the built tree
def num_nodes(root: 'MCTSNode') -> int:
    """Size of the tree."""
    num = 1
    for child in root.get_children().values():
        num += num_nodes(child)
    return num


def max_depth(root: 'MCTSNode') -> int:
    """Maximum depth of the tree."""
    depths = [max_depth(child) + 1 for child in root.get_children().values()]
    depth = max(depths) if len(depths) > 0 else 1
    return depth


def min_depth(root: 'MCTSNode') -> int:
    """Minimum depth of the tree."""
    depths = [min_depth(child) + 1 for child in root.get_children().values()]
    depth = min(depths) if len(depths) > 0 else 1
    return depth


class MCTSNode(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    """

    @abstractmethod
    def get_children(self) -> Dict[Tuple[int, int], 'MCTSNode']:
        # returns all children
        raise NotImplementedError()

    @abstractmethod
    def get_unexplored(self) -> Optional['MCTSNode']:
        # All possible action combinations that have not been explored yet
        raise NotImplementedError()

    @abstractmethod
    def get_total_reward(self) -> float:
        # total reward of a node
        raise NotImplementedError()

    @abstractmethod
    def incr_reward(self, reward) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_visit_count(self) -> int:
        # Total number of times visited this node (N)
        raise NotImplementedError()

    @abstractmethod
    def incr_visit_count(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def find_random_child(self) -> 'MCTSNode':
        # Random successor of this board state
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self) -> bool:
        # Returns True if the node has no children
        raise NotImplementedError()

    @abstractmethod
    def reward(self) -> float:
        # either reward or in our case the return value of the value function
        raise NotImplementedError()
