from collections import namedtuple, deque
import math
import random

import numpy as np

import gym
from gym.wrappers import RecordVideo

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils import Memory, Experience, ReplayDataset, DQNetwork

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepQLearner(pl.LightningModule):
    """
     Learner based on Deep Q-Network

    References: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, environment='LunarLander-v2', exploration_rate=0.99, n_episodes=int(1e3),
                 monitor=True,
                 discount_gamma=0.99,
                 use_experience_replay=True,
                 memory_size=int(1e6),
                 batch_size=10,
                 use_fixed_targets=True,
                 priority_rate=0.0,
                 **kwargs
                 ):
        """
        :param exploration_rate: float
            Epsilon used in \epsilon greedy strategy
        :param n_episodes: int
            Number of episodes which should agent perform
        :param monitor: bool
            Whether to record video of learning process.
            Note that this will assume that you have ffmpeg installed
            (on debian based distros sudo apt install ffmpeg will help
             have no idea about windows though :(
             )
        :param discount_gamma: float
            Discount factor
        :param environment: str
            Possible values are `Pong-ram-v4`, `CartPole-v1` ... see documentation of gym https://gym.openai.com/docs/
            Default value is `LunarLander-v2`
        :param use_experience_replay: bool
            Whether to use experience replay feature as introduced in https://arxiv.org/pdf/1312.5602.pdf
            If set to False then online learning will be used i.e. agent will use only current
            (state, action, next_state, reward) tuple as training sample
            Default is True
        :param memory_size: int
            Size of memory used for replaying experiences i.e. number of (state, action, next_state, reward) tuples
            to be stored in memory
            This will have effect only if `use_experience_replay=True`
        :param batch_size: int
            Size of mini-batch used in training step.
            If `use_experience_replay` parameter is set to False this will be set to 1
        :param use_fixed_targets: bool
            Whether to use two mirror neural networks one used for calculating target value of Q-function
            and other used for evaluating and updating weights
            Again introduced in https://arxiv.org/pdf/1312.5602.pdf
        :param priority_rate: float
            Exponent denoted as \alpha in paper https://arxiv.org/pdf/1511.05952.pdf determining how much priority
            give to each sample when using experience replay
        """
        super().__init__(**kwargs)
        if not use_experience_replay:  # this will ensure online learning
            memory_size = 1
            batch_size = 1
        self.experience = Memory(capacity=memory_size, priority_rate=priority_rate)
        self.batch_size = batch_size
        self.use_experience_replay = use_experience_replay
        self.use_fixed_targets = use_fixed_targets
        self.exploration_rate = exploration_rate
        self.n_episodes = n_episodes

        # the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes
        self.env = gym.make(environment)

        # q_model is Q function approximation
        self.q_model = DQNetwork(num_inputs=self.env.observation_space.shape[0],
                                 num_outputs=self.env.action_space.n)

        if use_fixed_targets:
            # this net will be used to calculate temporal difference error
            self.q_target_model = DQNetwork(num_inputs=self.env.observation_space.shape[0],
                                            num_outputs=self.env.action_space.n)
            self.q_target_model.load_state_dict(self.q_model.state_dict())
            self.q_target_model.eval()  # this sets self.training to False

        # Lightning trainer instance
        self.trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=0.5)
        if monitor:
            self.env = RecordVideo(self.env, environment)
        self.env.seed(42)

    def experience_replay(self):
        """
        We’ll be using experience replay memory for training our DQN.
         It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from
          it randomly, the transitions that build up a batch are decorrelated. It has been shown that this greatly
          stabilizes and improves the DQN training procedure.
          Why? Because it helps break the temporal correlation of training samples because neural nets work best with
          iid samples not correlated samples
        :return:
        """

    def random_policy(self):
        self.env.reset()
        score = 0
        for episode in range(self.n_episodes):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            score += reward
            if done:
                break
        self.env.close()
        return score

    def act(self, state, policy='random'):
        return self.env.action_space.sample() if (np.random.random() <= self.exploration_rate) else np.argmax(
            self.q_model.predict(state))

    def learn(self):
        trainer = pl.Trainer(accelerator="gpu", devices=1, strategy="dp", val_check_interval=100)

        trainer.fit(self.model)

        # override Lightning configure_optimizers() hook

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_size):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self):
        """
        Get train loader.
        """
        dataset = ReplayDataset(self.memory, self.episode_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)
        return dataloader


if __name__ == '__main__':
    m = Memory(10)
    m.push(1, 2, 3, 4)


