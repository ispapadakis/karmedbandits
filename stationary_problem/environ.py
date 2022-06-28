import numpy as np
import os
import pandas as pd

class RLEnviron:
    """
    K Armed Bandits
    Reinforcement Learning Environment
    """

    def __init__(self, bandits_file: str) -> None:
        """
        Initialize Bandits from .csv File
        """
        self.bandits_file = bandits_file
        self._bandits = []
        for _, row in pd.read_csv(bandits_file).iterrows():
            self.addBandit(Bandit(**row))

    def addBandit(self, bandit):
        """
        Add Bandit to Environment
        """
        if type(bandit) != Bandit:
            raise TypeError("Incorrect Bandit Type")
        self._bandits.append(bandit)

    def get_bandit(self, k:int):
        """
        Get Index k Bandit
        """
        return self._bandits[k]
        
    def get_size(self):
        """
        Get Number of Bandits in Enviroment
        """
        return len(self._bandits)

    def __repr__(self) -> str:
        """
        Representation of Class Member
        """
        return "RLEnviron(bandits_file='{0.bandits_file}')".format(self)

    def __str__(self) -> str:
        """
        Multiline Representation of Environment
        """
        out = "RL Environment:\n"
        for b in self._bandits:
            out += str(b) + "\n"
        return out

    def k_outcome(self, k:int):
        """
        Get (Potenially Random) Outcome from Kth Bandit
        """
        return self._bandits[k].get_reward()

    def opt_values(self):
        """
        True Bandit Values
        """
        return np.array([b.params['loc'] for b in self._bandits])

class Bandit:
    """
    Bandit With Normally Distributed Reward
    """

    def __init__(self, avg_reward:float, std_reward:float) -> None:
        self.params = {'loc':avg_reward, 'scale':std_reward}

    def get_reward(self):
        return np.random.normal(**self.params)

    def __repr__(self) -> str:
        fmt = "Bandit(avg_reward = {loc:}, std_reward = {scale:})"
        return fmt.format(**self.params)

class Policy:
    """
    Policy to Select Actions Given Environment
    """

    def __init__(
        self, 
        env: RLEnviron, 
        initial_value: float, 
        step_size: float = 1.0,
        step_type: str = 'default'
        ) -> None:

        if step_size <= 0.0 or step_size > 1.0:
            raise ValueError("step_size is out of range")
        self.env = env
        self.initial_value = initial_value
        self.Q = np.array([initial_value for _ in range(env.get_size())])
        self.step_size = step_size
        self.step_type = step_type

        self.bandit_counts = np.zeros(self.env.get_size(), dtype=np.int32)

    def select_action(self):
        """
        Select Random Option by Default
        """
        return np.random.choice(self.env.get_size())
    
    def q_update(self, k:int, reward:float):
        self.Q[k] += self.s_size(k) * (reward - self.Q[k])

    def update(self):
        action = self.select_action()
        reward = self.env.k_outcome(action)
        self.bandit_counts[action] += 1
        self.q_update(action, reward)
        return action, reward, self.Q

    def s_size(self, k:int):
        if self.step_type == 'constant':
            return self.step_size
        else:
            return self.step_size / self.bandit_counts[k]

    def __repr__(self) -> str:
        fmt = "Policy(env={0.env!r},initial_value={0.initial_value},step_size={0.step_size})"
        return fmt.format(self)

    def __str__(self) -> str:
        return "Random Action Policy"

class EpsilonGreedy(Policy):
    """
    Epsilon Greedy Policy Sub-Class
    """

    def __init__(
        self, 
        epsilon: float,
        env: RLEnviron, 
        initial_value: float, 
        step_size: float = 1
        ) -> None:
        super().__init__(env, initial_value, step_size)

        if self.env.get_size() < 1:
            raise RuntimeError("Environment is Empty")
        if epsilon <= 0 or epsilon > 1.0:
            raise ValueError("Epsilon is Out of Range")

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

    def __repr__(self) -> str:
        fmt = "EpsilonGreedy(epsilon={0.epsilon},env={0.env!r}"
        fmt += ",initial_value={0.initial_value},step_size={0.step_size})"
        return fmt.format(self)

    def __str__(self) -> None:
        return "Epsilon Greedy Policy eps={0.epsilon:.3f}".format(self)

class Greedy(Policy):
    """
    Greedy Policy Sub-Class
    """

    def select_action(self):
        max = self.Q.max()
        nearopt = np.where(self.Q > max - 1e-12)[0]
        return np.random.choice(nearopt)

    def __repr__(self) -> str:
        fmt = "Greedy(env={0.env!r},initial_value={0.initial_value},step_size={0.step_size})"
        return fmt.format(self)

    def __str__(self):
        return "Greedy Policy"


class UCB(Policy):
    """
    Upper-Confidence-Bound Policy Sub-Class
    """

    def __init__(
        self, 
        c_param: float,
        env: RLEnviron, 
        initial_value: float, 
        step_size: float = 1
        ) -> None:
        super().__init__(env, initial_value, step_size)

        if self.env.get_size() < 1:
            raise RuntimeError("Environment is Empty")
        if c_param <= 0:
            raise ValueError("c Parameter Non-Nositive")
        self.c_param = c_param

    def select_action(self):
        zero_count = np.where(self.bandit_counts==0)[0]
        if zero_count.size > 0:
            return np.random.choice(zero_count)

        t = self.bandit_counts.sum()
        crit = self.Q.copy()
        crit += self.c_param * np.sqrt(np.log(t)/self.bandit_counts)
        return np.argmax(crit)

    def __repr__(self) -> str:
        fmt = "UCB(c_param={0.c_param},env={0.env!r},"
        fmt += "initial_value={0.initial_value},step_size={0.step_size})"
        return fmt.format(self)
        
    def __str__(self) -> None:
        return "UCB Policy({0.c_param})".format(self)

def main():
    path = 'stationary_problem'
    file = 'bandits.csv'
    rlenv = RLEnviron(os.path.join(path,file))

    print(rlenv)

    print("\nTest repr rendering")
    print(repr(EpsilonGreedy(0.1,rlenv,0.0)))
    print("\nTest str rendering")
    print(EpsilonGreedy(0.1,rlenv,0.0))
    print("\n...\n")

    np.random.seed(2022)
    ucb_policy = UCB(0.1, rlenv, 5.0)

    print(ucb_policy)
    for i in range(10):
        action, reward, q = ucb_policy.update()
        fmt = "Action:{:2d} Reward:{:6.3f} Q:{}"
        print(fmt.format(action, reward, np.round(q,2)))

    print("Bandit Counts = ", ucb_policy.bandit_counts)


if __name__ == '__main__':
    main()