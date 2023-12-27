import time

from group02.game_state import game_state_from_obs, Agent
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
        self._init()

    def _init(self):
        self.me = Agent(0, 0, (0, 0), False, 1, [], 2, True)
        self.opponent = Agent(0, 0, (0, 0), False, 1, [], 2, True)
        self.prev_board = None
        self.tree = MCTS(None, 0, None, rollout_depth=7)
        self.action = None

    def act(self, obs, action_space):
        if obs['step_count'] == 0:
            self._init()

        # our agent id
        self.me.aid = self.agent_id
        self.opponent.aid = 1 - self.agent_id
        self.me.board_id = obs["alive"][self.me.aid]
        self.opponent.board_id = obs["alive"][self.opponent.aid]
        # it is not possible to use pommerman's forward model directly with observations,
        # therefore we need to convert the observations to a game state
        board, own_agent, opponent_agent, bombs, items, flames = game_state_from_obs(obs, self.me, self.opponent, self.prev_board)
        self.tree.action_space = action_space
        self.tree.agent_id = self.agent_id
        actions = (self.opponent.action, self.me.action) if self.me.aid > self.opponent.aid else (self.me.action, self.opponent.action)
        if not self.tree.step(actions):
            #print("RESETTING TREE " + str(obs['step_count']))
            self.tree.root_node = Node(board, own_agent, opponent_agent, bombs, items, flames, True)
        # TODO: if you can improve the approximation of the forward model (in 'game_state.py')
        #   then you can think of reusing the search tree instead of creating a new one all the time
        start_time = time.time()
        # now rollout tree for 450 ms
        #rollouts = 0
        while time.time() - start_time < 0.45:
            self.tree.do_rollout()
            #rollouts += 1
        #print(rollouts, "rollouts")
        self.me.action = self.tree.choose()
        self.prev_board = board
        return self.me.action
