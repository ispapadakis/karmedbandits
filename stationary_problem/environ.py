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

    @property
    def env_size(self):
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

    nearopt_epsilon = 1e-12

    def __init__(
        self, 
        env: RLEnviron, 
        initial_value: float, 
        step_size: float = 1.0,
        step_low_lim: float = None
        ) -> None:

        if step_size <= 0.0 or step_size > 1.0:
            raise ValueError("step_size is out of range")

        self.env = env
        if self.env.env_size < 1:
            raise RuntimeError("Environment is Empty")

        self.initial_value = initial_value
        self.Q = np.array([initial_value for _ in range(env.env_size)])
        self.step_size = step_size
        self.step_low_lim = step_low_lim
        self.bandit_counts = np.zeros(self.env.env_size, dtype=np.int32)
        self.prob = np.array([1.0/self.env.env_size for _ in range(self.env.env_size)]) 

    def select_action(self):
        """
        Select Random Option by Default
        """
        return np.random.choice(self.env.env_size, p=self.prob)
    
    def q_update(self, k:int, reward:float):
        self.Q[k] += self.s_size(k) * (reward - self.Q[k])

    def update_probs(self):
        """
        By Default Probabilities of Action Selection Are Kept Equal and Constant
        This Results in Random Action Policy
        """
        pass

    def update(self):
        self.update_probs()
        action = self.select_action()
        reward = self.env.k_outcome(action)
        self.bandit_counts[action] += 1
        self.q_update(action, reward)
        return action, reward

    def s_size(self, k:int):
        default_step_size = self.step_size / self.bandit_counts[k]
        if self.step_low_lim and (default_step_size < self.step_low_lim):
            return self.step_low_lim
        else:
            return default_step_size

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
        step_size: float = 1.0
        ) -> None:
        super().__init__(env, initial_value, step_size)

        if epsilon <= 0 or epsilon > 1.0:
            raise ValueError("Epsilon is Out of Range")

        self.epsilon = epsilon

    def update_probs(self):
        max = self.Q.max()
        opt = np.where(self.Q == max)[0]
        if self.env.env_size == len(opt):
            self.prob = np.array([1.0/self.env.env_size for _ in range(self.env.env_size)])
            return
        # Probability of Selection for Non Max Actions (Sums to epsilon)
        # Assign this probability by default
        n_non_max = self.env.env_size - len(opt)
        prob = np.ones(self.env.env_size) * self.epsilon / n_non_max
        # Probability of Selection for Max Actions (Sums to 1 - epsilon)
        prob[opt] = (1.0  - self.epsilon) / len(opt)
        prob = prob / prob.sum() # Assure Sum to 1.0
        self.prob = prob

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

    def update_probs(self):
        max = self.Q.max()
        nearopt = np.where(self.Q > max - self.nearopt_epsilon)[0]
        prob = np.zeros(self.env.env_size)
        prob[nearopt] = 1.0 / len(nearopt)
        self.prob = prob

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
        step_size: float = 1.0
        ) -> None:
        super().__init__(env, initial_value, step_size)

        if c_param <= 0:
            raise ValueError("c Parameter Has to Be Positive")
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

class GradientBandit(Policy):
    """
    Gradient Bandit Policy Sub-Class

    Q-Array Holds H-Array Values
    """

    def __init__(
        self, 
        env: RLEnviron, 
        h_step_size: float
        ) -> None:
        super().__init__(env, initial_value=0.0)

        self.average_reward = 0.0
        self.nreps = 0
        self.H = np.ones(self.env.env_size)
        self.h_step_size = h_step_size

    def update_probs(self):
        e = np.exp(self.H)
        self.prob = e / e.sum()

    def h_update(self, k:int, reward:float):
        rdiff = reward - self.average_reward
        self.H -= self.h_step_size * rdiff * self.prob
        self.H[k] += self.h_step_size * rdiff

    def update(self):
        action, reward = super().update()
        self.h_update(action, reward)
        self.nreps += 1
        self.average_reward +=  (reward - self.average_reward) / self.nreps
        return action, reward

    def __repr__(self) -> str:
        fmt = "GradientBandit(env={0.env!r},step_size={0.step_size})"
        return fmt.format(self)

    def __str__(self):
        return "Gradient Bandit Policy"


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

    np.random.seed(22)
    gb_policy = GradientBandit(rlenv, 0.1)
    print(gb_policy)
    print(repr(gb_policy))
    for i in range(10):
        action, reward = gb_policy.update()
        q = gb_policy.Q
        fmt = "Action:{:2d} Reward:{:6.3f} Avg Reward:{:6.3f} Q:{}"
        print(fmt.format(action, reward, gb_policy.average_reward, np.round(q,2)))

    print("Bandit Counts = ", gb_policy.bandit_counts)
    print("Selection Probabilities = ", gb_policy.prob)

    ra_policy = Policy(rlenv, 0.1, step_low_lim = 0.001)
    print(ra_policy)
    print(repr(ra_policy))
    for i in range(10):
        action, reward = ra_policy.update()
        q = ra_policy.Q
        fmt = "Action:{:2d} Reward:{:6.3f} Q:{}"
        print(fmt.format(action, reward, np.round(q,2)))

    print("Bandit Counts = ", ra_policy.bandit_counts)
    print("Selection Probabilities = ", ra_policy.prob)


if __name__ == '__main__':
    main()