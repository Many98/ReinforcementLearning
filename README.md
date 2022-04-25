# Repo for experiments with Reinforcment algortihms

#### Currently is implemented only Deep Q-Network

Solved can be any environment which is included in openai gym https://gym.openai.com/ and has 1-D state vector 

TODO add How to use ...


Implemented DQN was used for solving Pong environmnet (https://gym.openai.com/envs/Pong-ram-v0/) and 
 Lunar Lander environment (https://gym.openai.com/envs/LunarLander-v2/)


# Lunar Lander

There is dense reward so it was much more easy to solve this task for agent

Best Hyperparams used were:

batch_size: 64
dataset_size: 16384
discount_gamma: 0.99
environment: LunarLander-v2
exploration_decay_rate: 0.00013863
lr: 0.0001
memory_size: 100000
n_episodes: 1000
n_epochs: 500
n_most_recent: 10
penalization_type: discounted
reward_type: normal
start_exploration_rate: 1.0
stop_exploration_rate: 0.1
sync_rate: 5000
use_experience_replay: true
use_fixed_targets: true
use_prioritized_experiences: false
use_reward_shaping: false

# Pong

Pong results were unsatisfactory because I was unable to find optimal hyperparameters to dominate counter player.
This is probably because of very sparse rewards. As workaround was proposed gaussian redistribution of reward for when agent scores and discounted redistribution when
counter player scores. 
Maybe I there is needed more training...


Hyperparameters that yelded current best results were:

batch_size: 64
dataset_size: 1024
discount_gamma: 1.0
environment: Pong-ram-v4
exploration_decay_rate: 0.00013863
lr: 0.0001
memory_size: 10000
n_episodes: 1000
n_epochs: 500
n_most_recent: 10
penalization_type: discounted
reward_type: normal
start_exploration_rate: 1.0
stop_exploration_rate: 0.05
sync_rate: 5000
use_experience_replay: true
use_fixed_targets: true
use_prioritized_experiences: false
use_reward_shaping: true

