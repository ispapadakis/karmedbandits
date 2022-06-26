from environ import Policy, Greedy, EpsilonGreedy, RLEnviron, UCB
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_policy(policy, n):
    print(policy)
    actions = []
    rewards = []
    Q = []
    for _ in range(n):
        a, r, q = policy.update()
        actions.append(a)
        rewards.append(r)
        Q.append(q)
    return np.array(actions), np.array(rewards), np.asarray(Q), policy.bandit_counts

def summary_report(actions, rewards, Q, action_counts):
    print('- Average Reward: {:.3f}'.format(rewards.mean()))
    print(
        '- Final Q-values: [', 
        " ".join("{:.3f}".format(q) for q in Q[-1]),
        ']'
    )
    df = pd.DataFrame({'Action':actions, 'Reward':rewards})
    print('- Action Frequency: ',end='')
    for i,n in enumerate(action_counts):
        print("{}:{}".format(i,n), end=' ')
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
    eps_greedy = EpsilonGreedy(0.1, rlenv, 5.0)
    rnd_policy = Policy(rlenv, 5.0)
    ucb_policy = UCB(2.0, rlenv, 5.0)

    n_periods = 100000
    n_rolling = 1000
    n_warmup = 1000
    ewmalpha = 0.0001

    grd = summary_report(
        *simulate_policy(greedy, n_periods)
    )
 
    egrd = summary_report(
        *simulate_policy(eps_greedy, n_periods)
    )

    rpol = summary_report(
        *simulate_policy(rnd_policy, n_periods)
    )

    ucb = summary_report(
        *simulate_policy(ucb_policy, n_periods)
    )

    # FIGURE 1: EXP SMOOTH GRAPHS
    fig, ax = plt.subplots(figsize=(8, 6))
    grd['Reward'].rename(greedy).ewm(alpha=ewmalpha).mean()[n_warmup:].plot()
    egrd['Reward'].rename(eps_greedy).ewm(alpha=ewmalpha).mean()[n_warmup:].plot()
    ucb['Reward'].rename('UCB Policy').ewm(alpha=ewmalpha).mean()[n_warmup:].plot(ax=ax)    
    rpol['Reward'].rename('Random Policy').ewm(alpha=ewmalpha).mean()[n_warmup:].plot(ax=ax)    
    plt.title('Exp Smooth Means by Policy')
    plt.legend()
    fig.savefig(os.path.join(path, 'Figure_1.png'))
    plt.close()

    # FIGURE 2: MOVING AVG GRAPHS
    fig, ax = plt.subplots(figsize=(8, 6))
    grd['Reward'].rename(greedy).rolling(window=n_rolling).mean().plot(ax=ax)
    egrd['Reward'].rename(eps_greedy).rolling(window=n_rolling).mean().plot(ax=ax)
    ucb['Reward'].rename('UCB Policy').rolling(window=n_rolling).mean().plot(ax=ax)    
    rpol['Reward'].rename('Random Policy').rolling(window=n_rolling).mean().plot(ax=ax)
    plt.title(f'Rolling Means (mem={n_rolling}) by Policy')
    plt.legend()
    fig.savefig(os.path.join(path, 'Figure_2.png'))
    plt.close()

    # FIGURE 3: First Iterations
    fig, ax = plt.subplots(figsize=(8, 6))
    ucb['Action'][:100].plot(ax=ax, style='r-o', linewidth=.25, legend=True)
    ucb['Reward'][:100].plot(secondary_y=True,ax=ax, legend=True)
    plt.title("First 100 Iterations of UCB Policy")
    fig.savefig(os.path.join(path, 'Figure_3.png'))
    plt.close()


if __name__ == '__main__':
    np.random.seed(2022)
    main()