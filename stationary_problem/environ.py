import numpy as np
import os
import pandas as pd

class RLEnviron:
    def __init__(self, bandits_file: str) -> None:
        self._bandits = []
        for _, row in pd.read_csv(bandits_file).iterrows():
            self.addBandit(Bandit(**row))

    def addBandit(self, bandit):
        assert type(bandit) == Bandit
        self._bandits.append(bandit)

    def get_bandit(self, k:int):
        return self._bandits[k]
        
    def get_size(self):
        return len(self._bandits)

    def __repr__(self) -> str:
        out = "RL Environment:\n"
        for b in self._bandits:
            out += str(b) + "\n"
        return out

    def k_outcome(self, k:int):
        try:
            out = self._bandits[k].get_reward()
        except:
            raise
        return out

    def opt_values(self):
        return np.array([b.params['loc'] for b in self._bandits])

class Bandit:
    def __init__(self, avg_reward:float, std_reward:float) -> None:
        self.params = {'loc':avg_reward, 'scale':std_reward}

    def get_reward(self):
        return np.random.normal(**self.params)

    def __repr__(self) -> str:
        fmt = "Bandit(avg_reward = {loc:}, std_reward = {scale:})"
        return fmt.format(**self.params)

class Policy:
    def __init__(
        self, env: RLEnviron, 
        initial_value: float, 
        step_size: float = 1.0
        ) -> None:

        self.env = env
        self.Q = np.array([initial_value for _ in range(env.get_size())])
        self.step_size = step_size

    def select_action(self):
        """
        Select Random Option by Default
        """
        return np.random.choice(self.env.get_size())
    
    def q_update(self, k:int, reward:float):
        self.Q[k] += self.step_size * (reward - self.Q[k])

    def update(self):
        action = self.select_action()
        reward = self.env.k_outcome(action)
        self.q_update(action, reward)
        self.s_update()
        return action, reward, self.Q

    def s_update(self):
        n = 1.0 if self.step_size == 0.0 else 1.0 / self.step_size
        self.step_size = 1.0 / (n + 1.0)

class EpsilonGreedy(Policy):

    def __init__(
        self, 
        epsilon: float,
        env: RLEnviron, 
        initial_value: float, 
        step_size: float = 1
        ) -> None:
        super().__init__(env, initial_value, step_size)

        assert self.env.get_size() > 1
        assert epsilon > 0 and epsilon < 1.0
        self.epsilon = epsilon

        self.default_prob = np.ones(self.env.get_size()) 
        self.default_prob /= (self.env.get_size() - 1)
        self.default_prob *= self.epsilon

    def select_action(self):
        max = self.Q.max()
        opt = np.where(self.Q == max)[0]
        n = self.env.get_size()
        if n == len(opt):
            return np.random.choice(self.env.get_size())
        # Probability of Selection for Non Max Actions (Sums to epsilon)
        prob = np.ones(n) * self.epsilon / (n - len(opt))
        # Probability of Selection for Max Actions (Sums to 1 - epsilon)
        prob[opt] = (1.0  - self.epsilon) / len(opt)
        prob = prob / sum(prob) # Assure Sum to Zero
        return np.random.choice(self.env.get_size(), p=prob)
        
    def __repr__(self):
        return "Epsilon Greedy Policy eps={:.3f}".format(self.epsilon)

class Greedy(Policy):

    def select_action(self):
        max = self.Q.max()
        nearopt = np.where(self.Q > max - 1e-12)[0]
        return np.random.choice(nearopt)

    def __repr__(self):
        return "Greedy Policy"


def main():
    path = 'stationary_problem'
    file = 'bandits.csv'
    rlenv = RLEnviron(os.path.join(path,file))

    print(rlenv)

    eps_greedy = EpsilonGreedy(0.1, rlenv, 5.0)

    print(eps_greedy)
    for i in range(10):
        action, reward, q = eps_greedy.update()
        fmt = "Action:{:2d} Reward:{:6.3f} Q:{}"
        print(fmt.format(action, reward, np.round(q,2)))


if __name__ == '__main__':
    main()