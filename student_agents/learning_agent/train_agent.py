"""
This file shows a basic setup how a reinforcement learning agent
can be trained using DQN. If you are new to DQN, the code will probably be
not sufficient for you to understand the whole algorithm. Check out the
'Literature to get you started' section on the instruction sheet
if you want to have a look at additional resources.
Note that this basic implementation will not give a well performing agent
after training, but you should at least observe a small increase of reward.
"""

import torch
from torch.nn.functional import mse_loss
import random
import os
import numpy as np

from learning_agent import util
from learning_agent.net_input import featurize_simple
from learning_agent.net_architecture import DQN
from learning_agent.replay_memory import ReplayMemory, Transition


def select_action(policy_network: torch.nn.Module, device: torch.device, obs: torch.Tensor,
                  eps: float, n_actions: int) -> torch.Tensor:
    # choose a random action with probability 'eps'
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            # return action with highest q-value (expected reward of an action in a particular state)
            return policy_network(obs.to(device)).max(1)[1].view(1, 1).cpu()
    else:
        # return random action
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


def optimize_model(optimizer: torch.optim.Optimizer,
                   policy_network: torch.nn.Module,
                   target_network: torch.nn.Module,
                   device: torch.device,
                   replay_memory: ReplayMemory,
                   batch_size: int,
                   gamma: float) -> None:
    """
    This function updates the neural network.
    """
    # Sample a batch from the replay memory
    transitions = replay_memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # prepare the batch for further processing
    previous_states = torch.cat(batch.last_state).to(device)
    actions = torch.cat(batch.last_action).to(device)
    rewards = torch.cat(batch.reward).to(device)
    current_states = torch.cat(batch.current_state).to(device)
    terminal = torch.cat(batch.terminal).to(device)
    non_terminal = torch.tensor(tuple(map(lambda s: not s,
                                          batch.terminal)), device=device, dtype=torch.bool)

    # estimate q-values ( Q(s,a) ) by the policy network
    state_action_values = policy_network(previous_states).gather(1, actions)

    # estimate max_a' Q(s, a') by the target net
    # detach, because we do not need gradients here
    agent_reward_per_action = target_network(current_states).max(1)[0].detach()

    # calculating r + gamma * max_a' Q(s, a'), which serves as target value
    agents_expected_reward = torch.zeros(batch_size, device=device)
    # take only reward if it is a terminal step
    agents_expected_reward[terminal] = rewards[terminal]
    agents_expected_reward[non_terminal] = rewards[non_terminal] + \
        gamma * agent_reward_per_action[non_terminal].squeeze()

    # calculate loss
    loss = mse_loss(state_action_values, agents_expected_reward.unsqueeze(1))

    # set gradients to 0
    optimizer.zero_grad()
    # calculate new gradients
    loss.backward()
    # clip gradients
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    # perform the actual update step
    optimizer.step()


def train(device_name: str = "cuda",
          model_folder: str = "learning_agent/resources",
          model_file: str = "model.pt",
          load_model: bool = False,
          save_n_epochs: int = 100,
          target_update: int = 10,
          episodes: int = 10000,
          lr: float = 1e-3,
          memory_size: int = 100000,
          min_memory_size: int = 10000,
          render: bool = False,
          eps_start: float = 1.0,
          eps_end: float = 0.05,
          eps_dec: float = 0.00001,
          batch_size: int = 128,
          gamma: float = 0.99,
          print_n_epochs: int = 50) -> None:
    """
    Function to train a very simple DQN agent.

    Args:
        device_name: device used to train model
        model_folder: folder that model should be stored to
        model_file: file that model should be stored to
        load_model: specify if model should be loaded
        save_n_epochs: determines interval how often model is saved
        target_update: interval to update target network
        episodes: number of episodes to train
        lr: learning rate
        memory_size: maximum size of replay memory
        min_memory_size: minimum number of samples in replay memory to start optimization
        render: render environment graphically
        eps_dec: deduction rate of epsilon every epoch
        eps_start: initialize epsilon by value of 'eps_start'
        eps_end: decrease epsilon every epoch starting from 'eps_start' by 'eps_dec' until 'eps_end' is reached
        batch_size: batch size for a single update step of the model
        gamma: reward discount factor
        print_n_epochs: print stats every n epochs

    Returns:

    """
    device = torch.device(device_name)
    print("Running on device: {}".format(device))

    model_path = os.path.join(model_folder, model_file)

    # create the environment
    env, trainee, trainee_id, opponent, opponent_id = util.create_training_env()
    # resetting the environment returns observations for both agents
    state = env.reset()
    obs_trainee = state[trainee_id]
    obs_opponent = state[opponent_id]
    # featurize observations, such that they can be fed to a neural network
    obs_trainee_featurized = featurize_simple(obs_trainee)
    obs_size = obs_trainee_featurized.size()

    # create both the policy and the target network
    num_boards = obs_size[1]
    board_size = obs_size[2]
    policy_network = DQN(board_size=board_size, num_boards=num_boards, num_actions=env.action_space.n)
    policy_network.to(device)
    if load_model:
        print("Load model from path: {}".format(model_path))
        policy_network.load_state_dict(torch.load(model_path, map_location=device))
    target_network = DQN(board_size=board_size, num_boards=num_boards, num_actions=env.action_space.n)
    target_network.to(device)
    target_network.load_state_dict(policy_network.state_dict())

    # the optimizer is needed to calculate the gradients and update the network
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    # DQN is off-policy, it uses a replay memory to store transitions
    replay_memory = ReplayMemory(memory_size)

    episode_count = 0
    reward_count = 0
    # epsilon is needed to control the amount of exploration
    epsilon = eps_start

    # training loop
    while episode_count <= episodes:
        if render:
            env.render()

        # decrease epsilon over time
        if len(replay_memory) > min_memory_size and epsilon > eps_end:
            epsilon -= eps_dec
        action = select_action(policy_network, device, obs_trainee_featurized,
                               epsilon, env.action_space.n)

        # taking a step in the environment by providing actions of both agents
        actions = np.zeros(2)
        actions[trainee_id] = action.item()
        # getting action of opponent
        actions[opponent_id] = opponent.act(obs_opponent, env.action_space.n)
        current_state, reward, terminal, info = env.step(actions)
        obs_trainee_featurized_next = featurize_simple(current_state[trainee_id])

        # preparing transition (s, a, r, s', terminal) to be stored in replay buffer
        reward = float(reward[trainee_id])
        reward = torch.tensor([reward])
        terminal = torch.tensor([terminal], dtype=torch.bool)
        replay_memory.push(obs_trainee_featurized, action, reward, obs_trainee_featurized_next, terminal)

        # optimize model if minimum size of replay memory is filled
        if len(replay_memory) > min_memory_size:
            optimize_model(optimizer, policy_network, target_network, device,
                           replay_memory, batch_size, gamma)

        if terminal:
            episode_count += 1
            reward_count += reward.item()
            if render:
                env.render()
            env.close()

            # create new randomized environment
            env, trainee, trainee_id, opponent, opponent_id = util.create_training_env()
            state = env.reset()
            obs_trainee = state[trainee_id]
            obs_opponent = state[opponent_id]
            obs_trainee_featurized = featurize_simple(obs_trainee)

            if episode_count % save_n_epochs == 0:
                torch.save(policy_network.state_dict(), model_path)

            if episode_count % print_n_epochs == 0:
                print("Episode: {}, Reward: {}, Epsilon: {}, Memory Size: {}".format(
                    episode_count, reward_count, epsilon, len(replay_memory)))
                reward_count = 0

            # Update the target network, copying all weights and biases in DQN
            if episode_count % target_update == 0:
                target_network.load_state_dict(policy_network.state_dict())
        else:
            obs_trainee_featurized = obs_trainee_featurized_next
            obs_opponent = current_state[opponent_id]


if __name__ == "__main__":
    model = os.path.join("learning_agent", "resources")
    train(device_name='cpu', model_folder=model)
