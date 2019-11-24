"""
original: https://github.com/allanbreyes/gym-solutions
modifications from here: https://medium.com/datadriveninvestor/learning-machine-learning-roulette-with-monte-carlo-policy-db1b3b788230
"""

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import gym_casino

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys

# np.set_printoptions(threshold=sys.maxsize)

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def evaluate_rewards_and_transitions(problem, mutate=False):
    # Enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # Intiailize T and R matrices
    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    return R, T


@timing
def value_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10 ** 6, delta=10 ** -7):
    """ Runs Value Iteration on a gym problem """
    value_fn = np.zeros(problem.observation_space.n)
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)

    for i in range(max_iterations):
        previous_value_fn = value_fn.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        value_fn = np.max(Q, axis=1)

        if np.max(np.abs(value_fn - previous_value_fn)) < delta:
            break

    # Get and return optimal policy
    policy = np.argmax(Q, axis=1)
    return policy, i + 1


def encode_policy(policy, shape):
    """ One-hot encodes a policy """
    encoded_policy = np.zeros(shape)
    encoded_policy[np.arange(shape[0]), policy] = 1
    return encoded_policy


@timing
def policy_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10 ** 6, delta=10 ** -7):
    """ Runs Policy Iteration on a gym problem """
    num_spaces = problem.observation_space.n
    num_actions = problem.action_space.n

    # Initialize with a random policy and initial value function
    policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
    value_fn = np.zeros(num_spaces)

    # Get transitions and rewards
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)

    # Iterate and improve policies
    for i in range(max_iterations):
        previous_policy = policy.copy()

        for j in range(max_iterations):
            previous_value_fn = value_fn.copy()
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
            value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

            if np.max(np.abs(previous_value_fn - value_fn)) < delta:
                break

        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        policy = np.argmax(Q, axis=1)

        if np.array_equal(policy, previous_policy):
            break

    # Return optimal policy
    return policy, i + 1


def print_policy(policy, name, mapping=None, shape=(0,)):
    vis = np.array([mapping[action] for action in policy]).reshape(shape)
    print(vis)
    with open("output/{}.csv".format(name), 'w') as f:
        np.savetxt(f, vis, delimiter=',', fmt='%s')


def run_discrete(environment_name, mapping, shape=None, env_map=None):
    if env_map is None:
        problem = gym.make(environment_name)
    else:
        problem = gym.make(environment_name, desc=env_map, map_name=None)
    print('== {} =='.format(environment_name))
    print('Actions:', problem.env.action_space.n)
    print('States:', problem.env.observation_space.n)
    if environment_name != "Roulette-v0":
        print(environment_name)
        print(problem.env.desc)
    print()

    print('== Value Iteration ==')
    value_policy, iters = value_iteration(problem)
    print_policy(value_policy, "{}-value-it".format(environment_name), mapping, shape)
    print('Iterations:', iters)
    print()

    if shape is not None:
        print('== Value Policy ==')
        print_policy(value_policy, "{}-value-pol".format(environment_name), mapping, shape)

    print('== Policy Iteration ==')
    policy, iters = policy_iteration(problem)
    print('Iterations:', iters)
    print()

    diff = sum([abs(x - y) for x, y in zip(policy.flatten(), value_policy.flatten())])
    if diff > 0:
        print('Discrepancy:', diff)
        print()

    if shape is not None:
        print('== Iteration Policy ==')
        print_policy(policy, "{}-it-pol".format(environment_name), mapping, shape)
        print()

    return policy


# # FROZEN LAKE SMALL
mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
# shape = (4, 4)
# run_discrete('FrozenLake-v0', mapping, shape)
#
# # FROZEN LAKE LARGE
shape = (64,64)
random_map = generate_random_map(size=64, p=0.95)
run_discrete('FrozenLake8x8-v0', mapping, shape, random_map)
#
# # TAXI
# mapping = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
# run_discrete('Taxi-v3', mapping)

# Roulette
mapping = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
           10: "10", 11: "11", 12: "12", 13: "13", 14: "14", 15: "15", 16: "16", 17: "17", 18: "18", 19: "19",
           20: "20", 21: "21", 22: "22", 23: "23", 24: "24", 25: "25", 26: "26", 27: "27", 28: "28", 29: "29",
           30: "30", 31: "31", 32: "32", 33: "33", 34: "34", 35: "35", 36: "36", 37: "00"}
run_discrete('Casino-v0', mapping, shape=(1))


# TODO: compare the answers for value and policy iteration
# TODO: visualize the answers black=hole, blue=left, green=down, yellow=up, purple=down?