import torch
import os
import pkg_resources
from typing import Dict, Any
from gym import spaces

from pommerman import agents

from . import net_input
from . import net_architecture


# an example on how the trained agent can be used within the tournament
class LearningAgent(agents.BaseAgent):
    def __init__(self, *args, **kwargs):
        super(LearningAgent, self).__init__(*args, **kwargs)
        self.device = torch.device("cpu")  # you only have access to cpu during the tournament
        # place your model in the 'resources' folder and access them like shown here
        # change 'learning_agent' to the name of your own package (e.g. group01)
        data_path = pkg_resources.resource_filename('learning_agent', 'resources')
        model_file = os.path.join(data_path, 'model.pt')

        # loading the trained neural network model
        self.model = net_architecture.DQN(board_size=11, num_boards=7, num_actions=6)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()

    def act(self, obs: Dict[str, Any], action_space: spaces.Discrete) -> int:
        # the learning agent uses the neural net to find a move
        # the observation space has to be featurized before it is fed to the model
        obs_featurized: torch.Tensor = net_input.featurize_simple(obs).to(self.device)
        with torch.no_grad():
            action: torch.Tensor = self.model(obs_featurized).max(1)[1]  # take highest rated move
        return action.item()
