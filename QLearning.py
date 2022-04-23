import numpy as np

import gym
from gym.wrappers import RecordVideo

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils import Memory, Experience, ReplayDataset, DQNetwork, plot_learning_history

import os

import json

device = "cuda" if torch.cuda.is_available() else "cpu"


class DeepQLearner(pl.LightningModule):
    """
     Learner based on Deep Q-Network

    References: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, environment='Pong-ram-v4', start_exploration_rate=1.0, stop_exploration_rate=0.05,
                 exploration_decay_rate=0.000013863,
                 sync_rate=50,
                 lr=0.0001,
                 n_episodes=int(1e3),  # currently has no effect on training stopping use `n_epochs` instead
                 n_epochs=500,
                 monitor=True,
                 discount_gamma=1.0,
                 n_most_recent=50,
                 reward_type='normal',
                 penalization_type='linear',
                 use_experience_replay=True,
                 memory_size=int(1e4),
                 batch_size=64,
                 use_fixed_targets=True,
                 priority_rate=0.1,
                 use_prioritized_experiences=True,
                 dataset_size=1024,
                 use_reward_shaping=False,
                 path='',
                 **kwargs
                 ):
        """
        :param environment: str
            Possible values are `Pong-ram-v4`, `CartPole-v1` ... see documentation of gym https://gym.openai.com/docs/
            Default value is `LunarLander-v2`
        :param exploration_rate: float
            Epsilon used in epsilon greedy strategy
        :param exploration_decay_rate: float
            Decay factor used in exponential decay.
            New `exploration_factor` is calculated every new episode
            Default is 0.000013863 i.e. if `start_exploration_rate=1` then after 100000 steps (environment transitions)
             is `exploration_rate=0.25`
        :param stop_exploration_rate: float
            Stop value of `exploration_rate` denoted as epsilon in epsilon greedy strategy
        :param start_exploration_rate: float
            Start value of `exploration_rate` denoted as epsilon in epsilon greedy strategy
        :param n_episodes: int
            Number of episodes which should agent perform
            currently has no effect on training stopping use `n_epochs` instead
            TODO https://github.com/PyTorchLightning/pytorch-lightning/issues/1406
        :param n_epochs: int
            Number of epochs (iterations through dataset) should be performed to stop training.
            Usually there is more epochs than episodes
            Default is 500
        :param lr: float
            Learning rate for neural network
        :param monitor: bool
            Whether to record video of learning process.
            Note that this will assume that you have ffmpeg installed
            (on debian based distros sudo apt install ffmpeg will help
             have no idea about windows though :(
             )
        :param discount_gamma: float
            Discount factor
        :param use_experience_replay: bool
            Whether to use experience replay feature as introduced in https://arxiv.org/pdf/1312.5602.pdf
            If set to False then online learning will be used i.e. agent will use only current
            (state, action, next_state, reward) tuple as training sample.
            It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from
            it randomly, the transitions that build up a batch are decorelated. It has been shown that this greatly
            stabilizes and improves the DQN training procedure.
            Why? Because it helps break the temporal correlation of training samples because neural nets work best with
            iid samples not correlated samples
            Default is True
        :param memory_size: int
            Size of memory (`experience_buffer`) used for replaying experiences i.e. number of
             (state, action, next_state, reward) tuples to be stored in memory
            This will have effect only if `use_experience_replay=True`
        :param batch_size: int
            Size of mini-batch used in training step.
            If `use_experience_replay` parameter is False then this will be set to 1
        :param use_reward_shaping: bool
            Whether to use specific reward shaping which need to be explicitly implemented
        :param n_most_recent: int
            Number of most recent experiences where its reward should be shaped (changed)
        :param reward_type: str
            Type of reward shaping. Currently can be one of:
                `normal`, `discounted`, `exponential`, `uniform`, `linear`, `absolute`
            Note that `normal` and `absolute` types gives symmetric rewards around `n_most_recent // 2` i.e.
            Most recent and less recent experiences has smallest reward  and `n_most_recent // 2` experience
            will have highest reward
        :param penalization_type: str
            Type of cost shaping. "Opposite" to reward_type. Currently can be one of:
                `normal`, `discounted`, `exponential`, `uniform`, `linear`, `absolute`
            Note that `normal` and `absolute` types gives symmetric costs around `n_most_recent // 2` i.e.
            Most recent and less recent experiences has smallest cost  and `n_most_recent // 2` experience
            will have highest cost
        :param decreasing: bool
        :param sync_rate: int
            Defines how often synchronize `q_target_model` with `q_model`. `q_target_model` will be updates with
            weights of `q_model` every `sync_rate`th environment transition (training step)
        :param use_fixed_targets: bool
            Whether to use two mirror neural networks one used for calculating target value of Q-function
            and other used for evaluating and updating weights. Practically setting this parameter to False
            will set parameter `sync_rate` to 1.
            Again introduced in https://arxiv.org/pdf/1312.5602.pdf
        :param priority_rate: float
            Exponent denoted as alpha in paper https://arxiv.org/pdf/1511.05952.pdf determining how much priority
            give to each sample when using experience replay
            Default is 0.1
        :param use_prioritized_experiences: bool
            Whether to use prioritized experience buffer (Some experiences will be more likely sampled than others).
             If this parameter is set to False then `priority_rate=0` which
             will ensure uniform sampling
             Default is True
        :param dataset_size:
            Number of actions per epoch. This will define size of `ReplayDataset`
            One epoch is defined as one iteration through whole `ReplayDataset` i.e. there will be
            `n_actions_per_epoch / batch_size`
        :param path: str
            Absolute path where to store videos of training.
        """
        super().__init__(**kwargs)

        if not use_experience_replay:  # this will ensure online learning only with most recent experience tuple
            memory_size = 1
            batch_size = 1
        if not use_prioritized_experiences:
            priority_rate = 0.0

        self.experience_buffer = Memory(capacity=memory_size, priority_rate=priority_rate)
        self.batch_size = batch_size
        self.dataset_size = dataset_size  # size of replay dataset

        self.use_experience_replay = use_experience_replay
        self.use_fixed_targets = use_fixed_targets
        self.use_reward_shaping = use_reward_shaping

        self.lr = lr
        self.sync_rate = sync_rate

        self.start_exploration_rate = start_exploration_rate
        self.stop_exploration_rate = stop_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        self.n_episodes = n_episodes
        self.n_epochs = n_epochs

        self.discount_gamma = discount_gamma
        self.n_most_recent = n_most_recent
        self.reward_type = reward_type
        self.penalization_type = penalization_type

        # the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes
        self.env = gym.make(environment)

        if len(self.env.observation_space.shape) > 1:
            raise Exception(f'Observation space should be one-dimensional! \n'
                            f'Current environment has {len(self.env.observation_space.shape)} '
                            f'dimensional observation space with shape {self.env.observation_space.shape}')

        # q_model is Q function approximation
        self.q_model = DQNetwork(num_inputs=self.env.observation_space.shape[0],
                                 num_outputs=self.env.action_space.n)

        # this net will be used to calculate temporal difference error
        self.q_target_model = DQNetwork(num_inputs=self.env.observation_space.shape[0],
                                        num_outputs=self.env.action_space.n)
        self.q_target_model.load_state_dict(self.q_model.state_dict())
        self.q_target_model.eval()  # this sets self.training to False

        if not use_fixed_targets:
            self.sync_rate = 1  # this will ensure to update `q_target_model` every time `q_model` is updated

        if monitor:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # this is just patch for lunar_lander_v2
            self.env = RecordVideo(self.env, path, episode_trigger=lambda episode_id: \
                True if episode_id % 5 == 0 else False)
        self.env.seed(42)  # ensures reproducibility

        self.done = True  # indicator whether episode ended (=> we need reset environment)
        self.episode = -1
        self.state = self.reset_env()

        self.total_score = 0  # total score in episode
        self.episode_score = 0  # actual score in episode
        self.metrics = {'exploration': [], 'learning': [], 'epoch': []}

        self.path = path

        self.save_hyperparameters()
        self.explore(memory_size)  # initial exploration of environment by (agent)

    def reset_env(self):
        state = self.env.reset()
        self.done = False
        self.episode += 1
        return state

    def explore(self, num_steps):
        """
        Take `num_steps` random actions to explore environment and store experiences to `experience_buffer`
        :param num_steps: int
            Number of random steps in environment
        :return:
        """
        for step in range(num_steps):
            reward = self.step(exploration_rate=1.0)
            self.episode_score += reward
            if self.done:
                self.total_score = self.episode_score
                self.metrics['exploration'].append(self.episode_score)
                self.episode_score = 0

    def choose_action(self, exploration_rate):
        """
        Choose an action following epsilon greedy strategy.
        Note that actions always chooses network called `q_model`  responsible for acting.
        `q_target_model` is used for calculating best possible target q-value i.e. it represents
        value of target y in supervised setting (x, y)
        :param exploration_rate: float
            Exploration rate often denoted as epsilon in epsilon greedy strategy
        :return:
            (int, str)
            Action to be taken choose based on epsilon greedy strategy and type of action whether is random or optimal
        """
        np.random.seed(42)
        if np.random.random() <= exploration_rate:
            return self.env.action_space.sample(), 'random'
        else:
            q_val = self.q_model(torch.tensor(self.state).float().to(device))  # tensor of q-value for every action
            return int(q_val.argmax().item()), 'optimal'

    @torch.no_grad()
    def step(self, exploration_rate):
        """
        Take an action in environment and store experience into `experience_buffer`
        :param exploration_rate: float
        :return:
        """
        if self.done:
            self.state = self.reset_env()
            print(f' -----------------------\n'
                  f'Episode: {self.episode} \n'
                  f'Glob. step: {self.global_step} \n'
                  f'Total score per episode: {self.total_score} \n'
                  f'Exploration rate: {exploration_rate}'
                  )
            # every five episodes plot reward plot and save metrics
            if self.episode % 5 == 0:
                history_fig = plot_learning_history(metrics=self.metrics)
                history_fig.savefig(os.path.join(self.path, 'learning_history.png'))
                with open(os.path.join(self.path, 'metrics.json'), 'w') as json_file:
                    json.dump(self.metrics, json_file)

        action, action_type = self.choose_action(exploration_rate)

        next_state, reward, done, _ = self.env.step(action)

        experience = Experience(self.state, action, next_state, reward)

        # specific reward shaping, this setting should be used if reward is very sparse e.g. in pong
        if self.use_reward_shaping:
            if reward > 0:
                self.pong_reward_shaping(n_most_recent=self.n_most_recent, type=self.reward_type, decreasing=True,
                                         constant=1)
            elif reward < 0:
                self.pong_reward_shaping(n_most_recent=self.n_most_recent, type=self.penalization_type, decreasing=True,
                                         constant=-1)

        td_error = self.temporal_difference_error(reward, self.state, next_state, action_type, action)

        self.experience_buffer.push(experience, td_error)

        self.done = done
        self.state = next_state

        return reward

    def pong_reward_shaping(self, n_most_recent=50, type='discounted', decreasing=False, constant=1):
        """
        Specific reward shaping especially suitable for cases when reward is very sparse e.g. pong
        :param n_most_recent: int
            Number of most recent experiences where its reward should be shaped (changed)
        :param type: str
            Type of reward shaping. Currently can be one of:
                `normal`, `discounted`, `exponential`, `uniform`, `linear`, `absolute`
            Note that `normal` and `absolute` types gives symmetric rewards around `n_most_recent // 2` i.e.
            Most recent and less recent experiences has smallest reward (cost) and `n_most_recent // 2` experience
            will have highest reward (cost)
        :param decreasing: bool
            Whether should most recent rewards has higher reward (cost) than less recent
        :return:
        """
        if n_most_recent > len(self.experience_buffer):
            n_most_recent = len(self.experience_buffer) - 1
        for i in range(n_most_recent):
            state, action, next_state, reward = self.experience_buffer.memory[-i-1]
            if type == 'normal':
                shaped_reward = constant * np.exp(-((i - int(n_most_recent / 2)) ** 2) / (10 * n_most_recent))
            elif type == 'discounted':
                shaped_reward = constant * np.power(self.discount_gamma, i)
            elif type == 'exponential':
                if decreasing:
                    # value exponentially decreases from most recent experience i.e. most recent experience has highest reward
                    shaped_reward = constant * np.exp(-i / n_most_recent)
                else:
                    shaped_reward = constant * np.exp(-(n_most_recent - i) / n_most_recent)
            elif type == 'uniform':
                shaped_reward = constant * 1 / n_most_recent
            elif type == 'linear':
                if decreasing:
                    # value linearly decreases from most recent experience i.e. most recent experience has highest reward
                    shaped_reward = constant * (n_most_recent - i) / n_most_recent
                else:
                    shaped_reward = constant * i / n_most_recent
            elif type == 'absolute':
                # highest reward is in `int(n_most_recent / 2)`  (similar to normal)
                shaped_reward = constant * (1 - np.abs(i - (int(n_most_recent / 2)) / int(n_most_recent / 2)))
            else:
                shaped_reward = reward
            self.experience_buffer.memory[-i-1] = Experience(state, action, next_state, shaped_reward)

    def temporal_difference_error(self, reward, state, next_state, action_type, action):
        """
        Calculates temporal difference error
        :param reward: np.array
        :param state: np.array
        :param next_state: np.array
        :return:
        """
        if action_type == 'random':
            if len(self.experience_buffer) > 0:
                return np.max(self.experience_buffer.td_errors)
            else:
                return 0.5  # np.exp(reward)

        else:
            with torch.no_grad():
                q_value = self.q_model(torch.tensor(state).float().to(device)).detach().cpu().numpy()[action]
                next_state_value = self.q_target_model(
                    torch.tensor(next_state).float().to(device)).detach().cpu().numpy().max()

            target_q_value = next_state_value * self.discount_gamma + reward
            return np.abs(target_q_value - q_value)

    def loss(self, batch):
        """
        Creates a criterion that uses a squared term if the absolute
         element-wise error falls below beta and an L1 term otherwise.
          It is less sensitive to outliers than torch.nn.MSELoss and
           in some cases prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
        :param batch: tuple of torch.tensors

        :return:
        """
        states, actions, next_states, rewards = batch

        q_values = self.q_model(states.float()).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.q_target_model(next_states.float()).max(1)[0].detach()

        target_q_values = next_state_values * self.discount_gamma + rewards
        return torch.nn.SmoothL1Loss()(q_values, target_q_values)

    def forward(self, x):
        """
        Overrides lighting forward hook
        :param x: torch.Tensor
            Input state vector
        :return:
            Returns vector of q-values for every action
        """
        return self.q_model(x)

    # overrides lightning `configure_optimizers` hook
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.q_model.parameters(), lr=self.lr)
        #  optimizer = torch.optim.RMSprop(self.q_model.parameters(), lr=self.lr, alpha=0.99, eps=1e-8, weight_decay=0,
        #                               momentum=0)
        return [optimizer]

    # overrides lightning `training_step` hook
    def training_step(self, train_batch, batch_size):
        # exploration rate decay
        exploration_rate = max(self.stop_exploration_rate,
                               self.start_exploration_rate * np.exp(-self.exploration_decay_rate * self.global_step))

        # step through environment
        reward = self.step(exploration_rate)
        self.episode_score += reward

        loss = self.loss(train_batch)

        if self.done:
            self.total_score = self.episode_score
            self.metrics['learning'].append(self.episode_score)
            self.metrics['epoch'].append(self.current_epoch)
            self.episode_score = 0

        # every `self.sync_rate` update target neural network
        if self.global_step % self.sync_rate == 0:
            self.q_target_model.load_state_dict(self.q_model.state_dict())

        self.log("Episode_number", torch.tensor(self.episode).to(device), on_epoch=True, on_step=False,
                 prog_bar=True, logger=True)
        self.log("total_score_in_episode", torch.tensor(self.total_score).to(device), on_epoch=True, on_step=False,
                 prog_bar=True, logger=True)
        self.log("actual_score", torch.tensor(self.episode_score).to(device), on_epoch=True, prog_bar=True,
                 on_step=False, logger=True)
        self.log("steps performed", torch.tensor(self.global_step).to(device), on_epoch=True, on_step=False,
                 prog_bar=True, logger=True)

        return loss

    # overrides lightning `train_dataloader` hook
    def train_dataloader(self):
        """
        Get train loader.
        Train loader is created using `ReplayDataset` of stored experiences.
        """
        dataset = ReplayDataset(self.experience_buffer, sample_size=self.dataset_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)
        return dataloader

    def get_num_trainable_params(self):
        """
        Returns number of trainable parameters of neural network
        :return:
        """
        model_parameters = filter(lambda p: p.requires_grad, self.q_model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])


if __name__ == '__main__':
    # TODO try implementation with xgboost (note that incremental learning is not possible with gradient boosting)

    # learner = DeepQLearner.load_from_checkpoint(
    #    checkpoint_path="/home/fratrik/ReinforcementLearning/lightning_logs/version_2/checkpoints/epoch=999-step=50000.ckpt")

    learner = DeepQLearner(environment='Pong-ram-v4', path='/disk/fratrik/agents/pong/test_32_repeat_13',
                           start_exploration_rate=1.0,
                           stop_exploration_rate=0.05,
                           exploration_decay_rate=0.000013863,
                           sync_rate=5000,
                           discount_gamma=1.0,
                           use_reward_shaping=True,
                           n_most_recent=10,
                           reward_type='normal',
                           penalization_type='discounted',
                           use_prioritized_experiences=False,
                           priority_rate=0.2,

                           )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="/disk/fratrik/agents/pong/test_32_repeat_13",
                                                       save_top_k=2,
                                                       monitor="total_score_in_episode", mode="max")
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, precision=32, max_epochs=25000,
                         callbacks=[checkpoint_callback])
    trainer.fit(learner)

