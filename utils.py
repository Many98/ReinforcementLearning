import numpy as np
from collections import namedtuple, deque
from torch.utils.data.dataset import IterableDataset
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class DQNetwork(nn.Module):
    """
    Simple neural network approximator for Q-function.
    Input to neural network is not (state,action) pair but rather state.
    Output of network is approximation of Q-value for every action given input state
    """

    def __init__(self, num_inputs, num_outputs):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 64),  # `num_inputs` dimensional input state (Input layer)
            nn.ReLU(),
            nn.Linear(64, 128),  # Hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 128),  # Hidden layer 2
            nn.ReLU(),
            nn.Linear(128, 64),  # Hidden layer 3
            nn.ReLU(),
            nn.Linear(64, num_outputs),  # output is `num_outputs` Q-values for given state corresponding to actions
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
        # TODO maybe more optimal solution would be with priority queue which will pop experiences with small priority
        self.memory = deque(maxlen=capacity)
        self.td_errors = deque(maxlen=capacity)
        self.priority_rate = priority_rate  # this is denoted as \alpha in https://arxiv.org/pdf/1511.05952.pdf

    def push(self, experience, td_error):
        """

        :param experience: Experience
            Named tuple (state, action, next_state, reward)
        :param td_error: float
            Temporal difference error
        :return:
        """
        """
        Save a experience
        Note that `priority=0` will ensure to have every sample same probability to be sampled i.e. uniform distribution
        """
        self.memory.append(experience)
        self.td_errors.append(td_error)

    def sample(self, sample_size):
        # probabilities for each samples; if `priority_rate=0` then distribution will be uniform
        priority = np.power(np.array(self.td_errors) + 1e-5, self.priority_rate).astype('float64')
        probabilities = priority / np.sum(priority)
        indices = np.random.choice(len(self.memory),
                                   sample_size if sample_size < len(self.memory) else len(self.memory),
                                   replace=False, p=probabilities)
        states, actions, next_states, rewards = zip(*(self.memory[i] for i in indices))
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
    Iterable Dataset (for streaming data) containing the Memory (stream) which will be updated with new experiences
    during training.
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


def plot_learning_history(metrics):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    reward_exploration = [i for i in metrics['exploration']]
    reward_learning = [i for i in metrics['learning']]
    #epochs = [i for i in metrics['epoch']]
    #epochs = [0 for i in range(len(reward_exploration))] + epochs
    episodes = [i for i in range(len(reward_learning + reward_exploration))]

    ax.plot(episodes[:len(reward_exploration)], reward_exploration, linestyle='dashed')
    ax.plot(episodes[len(reward_exploration)-1:-1], reward_learning)
    #ax.xaxis.set_ticks(episodes)
    #locator = MaxNLocator(nbins=10)
    #ax.set_xticks
    #ax.set_xticklabels([(episode, epoch) for episode, epoch in zip(episodes, epochs)])
    #ax.tick_params(axis='x', labelrotation=45)

    ax.legend(['Initial exploration', 'Learning'], loc='upper right')

    ax.set_title('Agent learning')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episode')

    plt.tight_layout()

    return fig
