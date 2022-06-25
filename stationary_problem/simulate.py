from environ import Policy, Greedy, EpsilonGreedy, RLEnviron
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_policy(env, policy, n):
    print(policy)
    actions = []
    rewards = []
    Q = []
    for _ in range(n):
        a, r, q = policy.update()
        actions.append(a)
        rewards.append(r)
        Q.append(q)
    return np.array(actions), np.array(rewards), np.asarray(Q)

def main():
    path = 'stationary_problem'
    file = 'bandits.csv'
    rlenv = RLEnviron(os.path.join(path,file))

    print(rlenv)
    print("\nOptimal Values")
    print(rlenv.opt_values())

    greedy = Greedy(rlenv, 5.0)
    eps_greedy = EpsilonGreedy(0.01, rlenv, 5.0)

    n_periods = 100000
    n_rolling = 1000
    n_warmup = 1000
    ewmalpha = 0.001

    actions, rewards, Q = simulate_policy(rlenv, greedy, n_periods)
    print(rewards.mean())
    print(Q[-1])
    grd = pd.DataFrame({'Action':actions, 'Reward':rewards})
    print(grd['Action'].value_counts())
    #grd['Action'].plot()
    #grd['Reward'].plot(secondary_y=True)

    actions, rewards, Q = simulate_policy(rlenv, eps_greedy, n_periods)
    print(rewards.mean())
    print(Q[-1])
    egrd = pd.DataFrame({'Action':actions, 'Reward':rewards})
    print(egrd['Action'].value_counts())

    fig, ax = plt.subplots(figsize=(8, 6))

    grd['Reward'].rename(greedy).rolling(window=n_rolling).mean().plot(ax=ax)
    egrd['Reward'].rename(eps_greedy).rolling(window=n_rolling).mean().plot(ax=ax)
    plt.title(f'Rolling Means (mem={n_rolling}) by Policy')
    plt.legend()
    fig.savefig(os.path.join(path, 'Figure_2.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    grd['Reward'].rename(greedy).ewm(alpha=ewmalpha).mean()[n_warmup:].plot()
    egrd['Reward'].rename(eps_greedy).ewm(alpha=ewmalpha).mean()[n_warmup:].plot()
    plt.title('Exp Smooth Means by Policy')
    plt.legend()
    fig.savefig(os.path.join(path, 'Figure_1.png'))
    plt.close()


if __name__ == '__main__':
    np.random.seed(2022)
    main()