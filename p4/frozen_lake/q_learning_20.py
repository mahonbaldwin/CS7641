import gym
from gym import wrappers
import os
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map

import matplotlib.pyplot as plt

import time

s = 20
parent_step = None
step = 'ld-{}'.format(s)


def print_policy(policy, name):
    with open("output/{}.csv".format(name), 'w') as f:
        np.savetxt(f, policy, delimiter=',', fmt='%s')


# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def plot_it(rewards, episode=None):
    size = 5
    chunks = list(chunk_list(rewards, size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]
    plt.plot(range(0, len(rewards), size), averages)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    if episode is None:
        f_name = 'output/{}-base.png'.format(step)
    else:
        f_name = 'output/{}-{}-episode.png'.format(step, episode)
    plt.savefig(f_name, format='png')
    plt.clf()



start = time.time()
# Environment initialization
folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'q_learning')
folder2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

random_map = generate_random_map(size=s, p=0.95)
env = gym.wrappers.Monitor(gym.make('FrozenLake8x8-v0',desc=random_map, map_name=None), folder, force=True)

# Q and rewards
if parent_step is None:
    # Q = np.zeros((env.observation_space.n, env.action_space.n))
    Q = np.random.rand(env.observation_space.n, env.action_space.n) * 0.0001
else:
    Q = np.loadtxt("output/{}.csv".format(parent_step), delimiter=',')
rewards = []
iterations = []

# Parameters
alpha = 0.95
discount = 0.9999
episodes = 500000
# episodes = 5000

# Episodes
for episode in range(episodes):
    # Refresh state
    state = env.reset()
    done = False
    t_reward = 0
    max_steps = 50000
    # max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # Run episode
    for i in range(max_steps):
        if done:
            break

        current = state
        action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

    rewards.append(t_reward)
    iterations.append(i)

    if episode % 50000 == 0:
        print('{} - render chart'.format(episode))
        print_policy(Q, "{}-{}".format(step,str(int(time.time()))))
        plot_it(rewards, episode)
    elif episode % 5000 == 0:
        print(episode)

# Close environment
env.close()



print_policy(Q, step)

end = time.time()

print("{} µs".format((end - start) * 1000))

plot_it(rewards)
