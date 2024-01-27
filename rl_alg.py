import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import pandas as pd
import matplotlib.pyplot as plot

num_slices = 2
max_prbs_inf = 14
max_req_prbs = 2 # max total prbs a slice can require
min_req_prbs = 0
max_curr_prbs = max_req_prbs
min_curr_prbs = 0
max_allocate = max_req_prbs
min_allocate = -max_req_prbs

class RicEnv(Env):
    def __init__(self):
        # Here, the bounds are inclusive
        # action = [allocate_prbs_s1, allocate_prbs_s2]
        self.action_space = Box(low=np.array([min_allocate, min_allocate]), high=np.array([max_allocate, max_allocate]), dtype=int)
        # state = [prbs_inf, req_prbs_s1, req_prbs_s2, curr_prbs_s1, curr_prbs_s2]
        # (here, we're talking about total prbs)
        self.observation_space = Box(low=np.array([0, min_req_prbs, min_req_prbs, min_curr_prbs, min_curr_prbs]), high=np.array([max_prbs_inf, max_req_prbs, max_req_prbs, max_curr_prbs, max_curr_prbs]), dtype=int)

        # Creating function of req_prbs
        sine_time_range = np.arange(0, 10, 0.5)
        # req_prbs should oscillate around max_prbs_req/2
        self.sine_amplitude = np.rint((max_req_prbs/2) * np.sin(sine_time_range) + max_req_prbs/2)
        self.time = 0
        self.state = np.array([8, self.sine_amplitude[self.time], self.sine_amplitude[self.time], min_curr_prbs, min_curr_prbs])
        # remember: it's required prbs, not the amount that's there

    def calc_individual_reward(self, r, a):
        if a == r:
            individual_reward = 120
        elif a > r:
            individual_reward = 100 - (10 * (a - r))
        else:
            individual_reward = -100 + (10 * (r - a))

        return individual_reward

    def step(self, action):
        # save the current state
        prev_state = list(self.state)
        prbs_inf = prev_state[0]
        req_prbs_s1 = prev_state[1]
        req_prbs_s2 = prev_state[2]
        curr_prbs_s1 = prev_state[3]
        curr_prbs_s2 = prev_state[4]
        # update the state
        new_prbs_inf = self.state[0] - action[0] - action[1]
        new_curr_prbs_s1 = curr_prbs_s1 + action[0]
        new_curr_prbs_s2 = curr_prbs_s2 + action[1]
        self.time = self.time + 1  # update time
        if self.time > 19:
            self.time = 0 # reset time
        self.state = np.array([new_prbs_inf, self.sine_amplitude[self.time], self.sine_amplitude[self.time], new_curr_prbs_s1, new_curr_prbs_s2])

        # calculate reward
        # if not enough pRBs available
        if prbs_inf - (req_prbs_s1 - curr_prbs_s1) - (req_prbs_s2 - curr_prbs_s2) < 0:
            difference_s1 = req_prbs_s1 - curr_prbs_s1
            difference_s2 = req_prbs_s2 - curr_prbs_s2
            ideal_action_0 = round((difference_s1/(difference_s1+difference_s2-0.00001)) * prbs_inf)
            ideal_action_1 = round((difference_s2/(difference_s1+difference_s2+0.00001)) * prbs_inf)
            prev_state[1] = round(prev_state[3] + ideal_action_0)
            prev_state[2] = round(prev_state[4] + ideal_action_1)

        total_reward = 0
        individual_reward = 0
        for slice in range(1, num_slices+1):
            individual_reward = self.calc_individual_reward(prev_state[slice], self.state[slice+2])
            total_reward = total_reward + individual_reward
        reward = total_reward

        if new_prbs_inf < 0:
            reward = -1000
            self.state[0] = 0
        elif new_prbs_inf > max_prbs_inf:
            reward = -1000
            self.state[0] = max_prbs_inf

        if new_curr_prbs_s1 < 0:
            reward = -1000
            self.state[3] = 0
        elif new_curr_prbs_s1 > max_curr_prbs:
            reward = -1000
            self.state[3] = max_curr_prbs

        if new_curr_prbs_s2 < 0:
            reward = -1000
            self.state[4] = 0
        elif new_curr_prbs_s2 > max_curr_prbs:
            reward = -1000
            self.state[4] = max_curr_prbs


        info = {}
        done = False
        truncated = False
        return tuple(self.state), reward, done, truncated, info


# create environment
env = RicEnv()
state = tuple(env.state)

# initialize q table
# Here, the upper bound is exclusive
prbs_inf_states = np.arange(0, max_prbs_inf+1)
prbs_req_s1_states = np.arange(min_req_prbs, max_req_prbs+1)
prbs_req_s2_states = np.arange(min_req_prbs, max_req_prbs+1)
prbs_curr_s1_states = np.arange(min_curr_prbs, max_curr_prbs+1)
prbs_curr_s2_states = np.arange(min_curr_prbs, max_curr_prbs+1)
row_indices = pd.MultiIndex.from_product([prbs_inf_states, prbs_req_s1_states, prbs_req_s2_states, prbs_curr_s1_states, prbs_curr_s2_states])
action1_space = np.arange(min_allocate, max_allocate+1)
action2_space = np.arange(min_allocate, max_allocate+1)
col_indices = pd.MultiIndex.from_product([action1_space, action2_space])
q_values = np.zeros([np.size(prbs_inf_states)*np.size(prbs_req_s1_states)*np.size(prbs_req_s2_states)*np.size(prbs_curr_s1_states)*np.size(prbs_curr_s2_states),
                     np.size(action1_space)*np.size(action2_space)])
q_table = pd.DataFrame(q_values, columns=col_indices, index=row_indices)

# Hyperparameters
alpha = 0.99999
gamma = 0.00001
epsilon = 0.9999

# Training
for i in range(1, 400000):
    print(i)
    # select action (explore vs exploit)
    if random.uniform(0, 1) < epsilon:
        action = tuple(env.action_space.sample())  # Explore action space
    else:
        action = q_table.loc[state].idxmax()  # Exploit learned values

    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)
    # update q-value
    old_value = q_table.loc[state, action]
    next_max = q_table.loc[next_state].max()
    new_value = (1 - alpha) * old_value + alpha * (reward + (gamma * next_max))
    # new_value = reward # not using bellman equation
    q_table.loc[state, action] = new_value
    # update state
    state = next_state

q_table.to_csv('q_table.csv')

# Testing
# initializing the environment
env.time = 0
env.state = np.array([8, env.sine_amplitude[env.time], env.sine_amplitude[env.time], min_curr_prbs, min_curr_prbs])
state = tuple(env.state)
print("pRBs_inf \t pRBs_req \t pRBs_s1 \t pRBs_s2 \t A1 \t A2")
for i in range(0, 20): # upper bound is exclusive
    # select action (exploit)
    action = q_table.loc[state].idxmax()  # Exploit learned values
    print("{} \t \t {} \t \t {} \t \t {} \t \t {}  \t {}".format(env.state[0],
                                                                             env.sine_amplitude[env.time], env.state[3],
                                                                             env.state[4],
                                                                             action[0], action[1]))
    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)

    # update state
    state = next_state


