import time

from group02.game_state import game_state_from_obs
from group02.mcts import MCTS
from group02.node import Node
from pommerman import agents


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
