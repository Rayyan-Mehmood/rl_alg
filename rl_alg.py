import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import pandas as pd
import matplotlib.pyplot as plot

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
        self.state = np.array([10, self.sine_amplitude[self.time], self.sine_amplitude[self.time], min_curr_prbs, min_curr_prbs])
        # remember: it's required prbs, not the amount that's there

    def step(self, action):
        # save the current state
        prev_state = tuple(self.state)
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
        if prbs_inf - req_prbs_s1 - req_prbs_s2 < 0:
            difference_s1 = req_prbs_s1 - curr_prbs_s1
            difference_s2 = req_prbs_s2 - curr_prbs_s2
            ideal_action_0 = round((difference_s1/(difference_s1+difference_s2+0.00001)) * prbs_inf + 0.000001)
            ideal_action_1 = round((difference_s2/(difference_s1+difference_s2+0.00001)) * prbs_inf - 0.000001)
            if action[0] == ideal_action_0 and action[1] == ideal_action_1:
                reward = 1
            else:
                reward = -1
        # if enough pRBs available and slice gets the num prbs it requires, then reward
        elif new_curr_prbs_s1 == req_prbs_s1 and new_curr_prbs_s2 == req_prbs_s2:
            reward = 100
        # if enough pRBs available and slice is allocated more prbs than required
        elif new_curr_prbs_s1 > req_prbs_s1 and new_curr_prbs_s2 > req_prbs_s2:
            reward = 100 - 10 * ((new_curr_prbs_s1-req_prbs_s1)+(new_curr_prbs_s2-req_prbs_s2))
        # if enough pRBs available and slice 1 is allocated less prbs than required
        elif new_curr_prbs_s1 < req_prbs_s1:
            reward = - (req_prbs_s1/(new_curr_prbs_s1+0.00001))
        # if enough pRBs available and slice 2 is allocated less prbs than required
        else:
            reward = - (req_prbs_s2/(new_curr_prbs_s2+0.00001))

        if new_prbs_inf < 0:
            reward = -10
            self.state[0] = 0
        elif new_prbs_inf > max_prbs_inf:
            reward = -10
            self.state[0] = max_prbs_inf

        if new_curr_prbs_s1 < 0:
            reward = -10
            self.state[3] = 0
        elif new_curr_prbs_s1 > max_curr_prbs:
            reward = -10
            self.state[3] = max_curr_prbs

        if new_curr_prbs_s2 < 0:
            reward = -10
            self.state[4] = 0
        elif new_curr_prbs_s2 > max_curr_prbs:
            reward = -10
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
alpha = 0.7
gamma = 0.001
epsilon = 0.9999

# Training
for i in range(1, 200000):
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
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    # new_value = reward # not using bellman equation
    q_table.loc[state, action] = new_value
    # update state
    state = next_state

q_table.to_csv('q_table.csv')

# Testing
# initializing the environment
env.time = 0
env.state = np.array([10, env.sine_amplitude[env.time], env.sine_amplitude[env.time], min_curr_prbs, min_curr_prbs])
print("Inf pRBs \t Req pRBs \t Alloc pRBs S1 \t Alloc pRBs S2 \t pRBs S1 \t pRBs s2")
for i in range(0, 20): # upper bound is exclusive
    # select action (exploit)
    action = q_table.loc[state].idxmax()  # Exploit learned values
    print("{} \t \t {} \t \t {}   \t \t \t {}   \t \t \t {} \t \t {}".format(env.state[0], env.sine_amplitude[env.time], action[0], action[1], env.state[3], env.state[4]))
    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)
    # update state
    state = next_state


