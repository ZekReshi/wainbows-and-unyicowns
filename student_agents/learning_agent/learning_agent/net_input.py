import torch
import numpy as np
from typing import Dict, Any


def featurize_simple(obs: Dict[str, Any]) -> torch.Tensor:
    np_featurized: np.ndarray = _featurize_simple(obs)
    # convert to tensor and send to device
    return torch.tensor(np_featurized).unsqueeze(0)


def _featurize_simple(obs: Dict[str, Any]) -> np.ndarray:
    # here we encode the board observations into a structure that can
    # be fed into a convolution neural network
    board_size = len(obs['board'])

    # encoding consists of the following seven input planes:
    # 0. board representation
    # 1. bomb_blast_strength
    # 2. bomb_life
    # 3. bomb_moving_direction
    # 4. flame_life
    # 5. own position
    # 6. enemy position

    # board representation
    board_rep = obs['board'].astype(np.float32)

    # encode representation of bombs and flames
    bomb_blast_strength = obs['bomb_blast_strength'].astype(np.float32)
    bomb_life = obs['bomb_life'].astype(np.float32)
    bomb_moving_direction = obs['bomb_moving_direction'].astype(np.float32)
    flame_life = obs['flame_life'].astype(np.float32)

    # encode position of trainee
    position = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    position[obs['position'][0], obs['position'][1]] = 1.0

    # encode position of enemy
    enemy = obs['enemies'][0]  # we only have to deal with 1 enemy
    enemy_position = np.where(obs['board'] == enemy.value, 1.0, 0.0).astype(np.float32)

    # stack all the input planes
    featurized = np.stack((board_rep, bomb_blast_strength, bomb_life, bomb_moving_direction,
                           flame_life, position, enemy_position), axis=0)
    return featurized
