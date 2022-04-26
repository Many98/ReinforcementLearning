# Repo for experiments with Reinforcment algortihms

#### Currently is implemented only Deep Q-Network

Implementation of Policy Nework is planned

Solved can be any environment which is included in openai gym https://gym.openai.com/ and has 1-D state vector 

`TODO add How to use ...`


Implemented DQN was used for solving Pong environmnet (https://gym.openai.com/envs/Pong-ram-v0/) and 
Lunar Lander environment (https://gym.openai.com/envs/LunarLander-v2/)


# Pong

Pong results were unsatisfactory as I was unable to find optimal hyperparameters to dominate counter player.
This is probably because of very sparse rewards as agent receives +1 point when he scores and -1 point when counter player scores. Therefore it is difficult for agent to find out which sequence of actions led to desired or undesired outcome. 
As workaround was proposed quasi gaussian redistribution of reward `reward_type='normal'` when agent scores and discounted redistribution `penalization_type='discounted'` controlled with parameter `discount_gamma` when counter player scores. Redistribution was applied for `n_most_recent` steps. 

Parameter `sync_rate` proved itself very important as it controls after how many steps should target network synchronize its weights with behavioral network. After few trials and errors with very low values of this parameter 50 - 200, I found out that optimal value is like thousands instead of tens or hundreds. 

Maybe agent requires more training...

#### Learning curve of current best agent

We can see that agent was able hardly achieve average score of -15 after learning for about 175 episodes
![learning_history_pong](https://user-images.githubusercontent.com/65658910/165339072-45c11fca-7aea-425e-860d-28acd2775bbc.png)


Hyperparameters that yielded current best results were:

- environment: Pong-ram-v4
- batch_size: 64
- dataset_size: 1024
- memory_size: 10000
- use_experience_replay: true
- use_prioritized_experiences: false
- discount_gamma: 1.0
- lr: 0.0001
- use_reward_shaping: true
- n_most_recent: 10
- penalization_type: discounted
- reward_type: normal
- start_exploration_rate: 1.0
- stop_exploration_rate: 0.05
- exploration_decay_rate: 0.00013863
- use_fixed_targets: true
- sync_rate: 5000



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


# Lunar Lander


There is dense reward so it was much more easy to solve this task for agent

Best Hyperparams used were:

- environment: LunarLander-v2
- batch_size: 64
- dataset_size: 16384
- memory_size: 100000
- use_experience_replay: true
- use_prioritized_experiences: false
- discount_gamma: 0.99
- lr: 0.0001
- use_reward_shaping: false
- n_most_recent: 10
- penalization_type: discounted
- reward_type: normal
- start_exploration_rate: 1.0
- stop_exploration_rate: 0.1
- exploration_decay_rate: 0.00013863
- use_fixed_targets: true
- sync_rate: 5000



### Best achieved results

#### Learning history of an agent 

![learning_history](https://user-images.githubusercontent.com/65658910/165336279-f2c9b4cd-d06b-46f8-93bc-7de27784180f.png)


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


