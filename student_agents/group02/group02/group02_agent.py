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
        self.me = Agent(0, 0, (0, 0), False, 1, 1, 1, True)
        self.opponent = Agent(0, 0, (0, 0), False, 1, 1, 1, True)
        self.prev_board = None

    def act(self, obs, action_space):
        # our agent id
        self.me.aid = self.agent_id
        self.opponent.aid = 1 - self.agent_id
        self.me.board_id = obs["alive"][self.me.aid]
        self.opponent.board_id = obs["alive"][self.opponent.aid]
        # it is not possible to use pommerman's forward model directly with observations,
        # therefore we need to convert the observations to a game state+
        board, own_agent, opponent_agent, bombs, items, flames = game_state_from_obs(obs, self.me, self.opponent, self.prev_board)
        root = Node(board, own_agent, opponent_agent, bombs, items, flames)
        # TODO: if you can improve the approximation of the forward model (in 'game_state.py')
        #   then you can think of reusing the search tree instead of creating a new one all the time
        tree = MCTS(action_space, self.agent_id, (board, own_agent, opponent_agent, bombs, items, flames), rollout_depth=5)  # create tree
        start_time = time.time()
        # now rollout tree for 450 ms
        while time.time() - start_time < 0.45:
            tree.do_rollout(root)
        move = tree.choose(root)
        self.prev_board = board
        return move
