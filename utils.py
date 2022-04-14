import numpy as np
from collections import namedtuple, deque
from torch.utils.data.dataset import IterableDataset
import torch.nn as nn


class DQNetwork(nn.Module):
    """
    Simple neural network approximator for Q-function.
    Input to neural network is not (state,action) pair but rather state.
    Output of network is approximation of Q-value for every action given input state
    """

    def __init__(self, num_inputs, num_outputs):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 512),  # `num_inputs` dimensional input state (Input layer)
            nn.ReLU(),
            nn.Linear(512, 512),  # Hidden layer
            nn.ReLU(),
            nn.Linear(512, num_outputs),  # output is `num_outputs` Q-values for given state corresponding to actions
        )

    def forward(self, x):
        return self.net(x)


Experience = namedtuple('Transition',
                        field_names=('state', 'action', 'next_state', 'reward'))


class Memory(object):
    """
    Class representing memory as (state, action, next_state, reward) tuples i.e. already gained experience of agent.
    It allows to take random samples which will help agent to use data samples (experiences) in more efficient way
    and decrease correlation between samples when compared to using vanilla online learning. For details of memory
    replay see original paper from DeepMind https://arxiv.org/pdf/1312.5602.pdf

    Class also implements prioritized sampling instead of simple uniform sampling.
    Priority rate is based on Temporal difference error. For details see https://arxiv.org/pdf/1511.05952.pdf


    Code for this class was inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity, priority_rate=0.0):
        # note that deque will automatically pop old experiences and add new when its full
        self.memory = deque(maxlen=capacity)
        self.td_errors = deque(maxlen=capacity)
        self.priority_rate = priority_rate  # this is denoted as \alpha in https://arxiv.org/pdf/1511.05952.pdf

    def push(self, state, action, next_state, reward, td_error):
        """
        Save a experience
        Note that `priority=0` will ensure to have every sample same probability to be sampled i.e. uniform distribution
        """
        self.memory.append(Experience(state, action, next_state, reward))
        self.td_errors.append(td_error)

    def sample(self, batch_size):
        # probabilities for each samples; if `priority_rate=0` then distribution will be uniform
        probabilities = np.power(np.array(self.td_errors) + 1e-5, self.priority_rate) / np.sum(self.td_errors)
        samples = np.random.choice(self.memory, batch_size, replace=False, p=probabilities)
        states, actions, next_states, rewards = zip(*(sample for sample in samples))
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards, dtype=np.float32),

        )

    def __len__(self):
        return len(self.memory)


class ReplayDataset(IterableDataset):
    """
    Iterable Dataset containing the Memory which will be updated with new experiences during training.
    """

    def __init__(self, memory, sample_size=200):
        """
        Args:
            memory: Memory
            sample_size: number of experiences to sample at a time
        """
        self.memory = memory
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, new_states, rewards = self.memory.sample(self.sample_size)
        for i in range(len(rewards)):
            yield states[i], actions[i], new_states[i], rewards[i]
