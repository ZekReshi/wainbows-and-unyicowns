import time
from typing import Dict, Any
from gym import spaces

from pommerman import agents
from .game_state import game_state_from_obs
from .node import Node
from .mcts import MCTS


class HeuristicAgent(agents.BaseAgent):
    def __init__(self, *args, **kwargs):
        super(HeuristicAgent, self).__init__(*args, **kwargs)

    def act(self, obs: Dict[str, Any], action_space: spaces.Discrete) -> int:
        # our agent id
        agent_id = self.agent_id
        # it is not possible to use pommerman's forward model directly with observations,
        # therefore we need to convert the observations to a game state
        game_state = game_state_from_obs(obs)
        root = Node(game_state, agent_id)
        root_state = root.state  # root state needed for value function
        # TODO: if you can improve the approximation of the forward model (in 'game_state.py')
        #   then you can think of reusing the search tree instead of creating a new one all the time
        tree = MCTS(action_space, agent_id, root_state)  # create tree
        start_time = time.time()
        # now rollout tree for 450 ms
        while time.time() - start_time < 0.45:
            tree.do_rollout(root)
        move = tree.choose(root)
        return move
