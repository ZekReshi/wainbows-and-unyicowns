import random
from collections import namedtuple
from typing import List

Transition = namedtuple('Transition',
                        ('last_state', 'last_action', 'reward', 'current_state', 'terminal'))


class ReplayMemory(object):
    """
    This class saves transitions that are used for optimization.
    """
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.memory: List[Transition] = []
        self.position: int = 0

    def push(self, *args) -> None:
        # save a transition
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        # sample a batch uniformly from memory
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
