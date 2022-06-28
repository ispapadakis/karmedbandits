# Comparing Greedy and Eps-Greedy Policies

## RL Environment:
Bandit(avg_reward = 1.5, std_reward = 0.1)
Bandit(avg_reward = 2.0, std_reward = 0.2)
Bandit(avg_reward = 0.0, std_reward = 1.0)


Optimal Values: [1.5 2.  0. ]
## Greedy Policy
- Average Reward: 2.001
- Final Q-values: [ 1.512 2.001 1.033 ]
- Action Frequency: 0:1 1:99998 2:1 
## Epsilon Greedy Policy eps=0.100
- Average Reward: 1.874
- Final Q-values: [ 1.501 2.000 0.011 ]
- Action Frequency: 0:4941 1:89961 2:5098 
## UCB Policy(2.0)
- Average Reward: 1.999
- Final Q-values: [ 1.489 2.000 0.169 ]
- Action Frequency: 0:163 1:99823 2:14 
## Random Action Policy
- Average Reward: 1.170
- Final Q-values: [ 1.500 2.000 -0.004 ]
- Action Frequency: 0:33266 1:33611 2:33123 
![plot](Figure_3.png)
![plot](Figure_1.png)
![plot](Figure_2.png)
