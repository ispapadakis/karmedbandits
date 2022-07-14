# karmedbandits
Reinforcement Learning Examples

## [Stationary Bandits](stationary_problem/Readme.md)

**Comparison of Policies in Terms of Environment Exploration and Immediate Reward Exploitation.**

Performance in range from 1 to 3 (1 is best).

Exploration effectiveness is measured by how well average bandit rewards come to actual mean bandit rewards. Exploitation effectiveness is measured by how close average reward is to maximum mean bandit reward. 

| Policy                 | Exploration | Exploitation |
| :---                   | :---:       | :---:        |
| Greedy Policy          | :three:     |:one:         |
| Epsilon Greedy Policy  | :one:       |:three:       |
| UCB Policy             | :three:     |:two:         |
| Gradient Bandit Policy | :two:       |:two:         |
| Random Action Policy   | :one:       |              |

- :white_check_mark: Gradient Bandit Policy appears to offer the best of both worlds in this simulation.

- UCB Policy has similarly good performance.

- Epsilon-Greedy Policy (with epsilon = 0.1) appears to do more exploration than necessary for this setting.
