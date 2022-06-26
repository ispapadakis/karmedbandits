# Comparing Greedy and Eps-Greedy Policies

## RL Environment
- Bandit(avg_reward = 1.5, std_reward = 0.1)
- Bandit(avg_reward = 2.0, std_reward = 0.2)
- Bandit(avg_reward = 0.0, std_reward = 1.0)

Optimal Values: [1.5, 2.0, 0.0]


## Greedy Policy

- Average Reward: 2.000
- Final Q-values: [ 1.989 2.001 1.970 ]
- Action Frequency: 0:16 1:99978 2:6 

## Epsilon Greedy Policy (eps=0.1)

- Average Reward: 1.874
- Final Q-values: [ 1.921 2.001 -0.030 ]
- Action Frequency: 0:4999 1:89903 2:5098 

## UCB Policy

- Average Reward: 1.995
- Final Q-values: [ 1.730 1.999 1.139 ]
- Action Frequency: 0:547 1:99393 2:60 

## Random Action Policy

- Average Reward: 1.170
- Final Q-values: [ 1.503 2.106 0.170 ]
- Action Frequency: 0:33169 1:33513 2:33318 

![plot](Figure_3.png)
![plot](Figure_1.png)
![plot](Figure_2.png)
