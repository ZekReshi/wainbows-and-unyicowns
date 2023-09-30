from . import BaseAgent
from pommerman.constants import Action


# does nothing else than sending a Stop move
class DummyAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(DummyAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        return Action.Stop
