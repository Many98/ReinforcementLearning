# Repo for experiments with Reinforcment algortihms

#### Currently is implemented only Deep Q-Network

Solved can be any environment which is included in openai gym https://gym.openai.com/ and has 1-D state vector 

TODO add How to use ...


Implemented DQN was used for solving Pong environmnet (https://gym.openai.com/envs/Pong-ram-v0/) and 
 Lunar Lander environment (https://gym.openai.com/envs/LunarLander-v2/)


# Lunar Lander


There is dense reward so it was much more easy to solve this task for agent

Best Hyperparams used were:

- batch_size: 64
- dataset_size: 16384
- discount_gamma: 0.99
- environment: LunarLander-v2
- exploration_decay_rate: 0.00013863
- lr: 0.0001
- memory_size: 100000
- n_episodes: 1000
- n_epochs: 500
- n_most_recent: 10
- penalization_type: discounted
- reward_type: normal
- start_exploration_rate: 1.0
- stop_exploration_rate: 0.1
- sync_rate: 5000
- use_experience_replay: true
- use_fixed_targets: true
- use_prioritized_experiences: false
- use_reward_shaping: false

### Best achieved results

#### After 100 episodes (just exploration)

https://user-images.githubusercontent.com/65658910/165178406-2f697b7a-43d9-4174-82ce-446d2dcb0985.mp4

#### After 1000 episodes (still only initial exploration :D)

https://user-images.githubusercontent.com/65658910/165178542-38dd3270-a9ca-4ab0-b880-280254258f5d.mp4

#### Episode 1410 (finally some learning)

https://user-images.githubusercontent.com/65658910/165179108-4ff31c03-7603-4258-87e3-f2ec59555859.mp4



#### Episode 1850)

https://user-images.githubusercontent.com/65658910/165179263-61fb075a-c05b-4ffb-b4d7-58f8b72a3b10.mp4


#### Episode 2035
https://user-images.githubusercontent.com/65658910/165179276-b8622348-db29-49e8-bd04-c520c9b45009.mp4



#### Episode 10000
https://user-images.githubusercontent.com/65658910/165179316-ba426445-268d-49ec-9713-0bdc0f09c8e9.mp4



# Pong

Pong results were unsatisfactory as I was unable to find optimal hyperparameters to dominate counter player.
This is probably because of very sparse rewards. As workaround was proposed gaussian redistribution of reward for when agent scores and discounted redistribution when
counter player scores. 
Maybe I there is needed more training...


Hyperparameters that yelded current best results were:

- batch_size: 64
- dataset_size: 1024
- discount_gamma: 1.0
- environment: Pong-ram-v4
- exploration_decay_rate: 0.00013863
- lr: 0.0001
- memory_size: 10000
- n_episodes: 1000
- n_epochs: 500
- n_most_recent: 10
- penalization_type: discounted
- reward_type: normal
- start_exploration_rate: 1.0
- stop_exploration_rate: 0.05
- sync_rate: 5000
- use_experience_replay: true
- use_fixed_targets: true
- use_prioritized_experiences: false
- use_reward_shaping: true

#### Episode 25

https://user-images.githubusercontent.com/65658910/165179811-f6119ce0-ad6b-4de1-8913-03f37672e25b.mp4



#### Episode 50

https://user-images.githubusercontent.com/65658910/165179829-bae4c822-009c-48cd-b6b7-6048e4b707ac.mp4



#### Episode 75

https://user-images.githubusercontent.com/65658910/165179845-b9f5c621-5a9e-43a8-a448-625225cb4977.mp4


#### Episode 100

https://user-images.githubusercontent.com/65658910/165179854-149aa75b-8521-475e-ba4e-fdc18fd7fac7.mp4


#### Episode 125 Only one episode when agent managed to win :(

https://user-images.githubusercontent.com/65658910/165179878-cd27cf83-5d08-4f9e-8a5c-4b3ec0274fed.mp4


#### Episode 135

https://user-images.githubusercontent.com/65658910/165179885-b7f4fd88-9963-46c8-bf20-80e0c08e3379.mp4

