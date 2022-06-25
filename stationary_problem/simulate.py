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

def summary_report(actions, rewards, Q):
    print('- Average Reward: {:.3f}'.format(rewards.mean()))
    print(
        '- Final Q-values: [', 
        " ".join("{:.3f}".format(q) for q in Q[-1]),
        ']'
    )
    df = pd.DataFrame({'Action':actions, 'Reward':rewards})
    print('- Action Frequency: ',end='')
    for t in df['Action'].value_counts().items():
        print(t, end=' ')
    print()
    return df

def main():
    path = 'stationary_problem'
    file = 'bandits.csv'
    rlenv = RLEnviron(os.path.join(path,file))

    print(rlenv)
    print("\nOptimal Values")
    print(rlenv.opt_values())

    greedy = Greedy(rlenv, 5.0)
    eps_greedy = EpsilonGreedy(0.01, rlenv, 5.0)
    rnd_policy = Policy(rlenv, 5.0)

    n_periods = 100000
    n_rolling = 1000
    n_warmup = 1000
    ewmalpha = 0.001

    grd = summary_report(
        *simulate_policy(rlenv, greedy, n_periods)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    grd['Action'][:100].plot(kind='hist', ax=ax)
    grd['Reward'][:100].plot(secondary_y=True,ax=ax)
    plt.legend()
    plt.title("First 100 Iterations of Greedy Policy")
    fig.savefig(os.path.join(path, 'Figure_3.png'))
    plt.close()

 
    egrd = summary_report(
        *simulate_policy(rlenv, eps_greedy, n_periods)
    )

    rpol = summary_report(
        *simulate_policy(rlenv, rnd_policy, n_periods)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    grd['Reward'].rename(greedy).rolling(window=n_rolling).mean().plot(ax=ax)
    egrd['Reward'].rename(eps_greedy).rolling(window=n_rolling).mean().plot(ax=ax)
    rpol['Reward'].rename('Random Policy').rolling(window=n_rolling).mean().plot(ax=ax)
    plt.title(f'Rolling Means (mem={n_rolling}) by Policy')
    plt.legend()
    fig.savefig(os.path.join(path, 'Figure_2.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    grd['Reward'].rename(greedy).ewm(alpha=ewmalpha).mean()[n_warmup:].plot()
    egrd['Reward'].rename(eps_greedy).ewm(alpha=ewmalpha).mean()[n_warmup:].plot()
    rpol['Reward'].rename('Random Policy').ewm(alpha=ewmalpha).mean().plot(ax=ax)    
    plt.title('Exp Smooth Means by Policy')
    plt.legend()
    fig.savefig(os.path.join(path, 'Figure_1.png'))
    plt.close()


if __name__ == '__main__':
    np.random.seed(2022)
    main()