import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import pandas as pd
import matplotlib.pyplot as plot

max_prbs_inf = 25
max_req_prbs = 2 # max_req_prbs = max_action
min_req_prbs = -2

class RicEnv(Env):
    def __init__(self):
        # Here, the bounds are inclusive
        self.action_space = Box(low=np.array([min_req_prbs, min_req_prbs]), high=np.array([max_req_prbs, max_req_prbs]), dtype=int)
        self.observation_space = Box(low=np.array([0, min_req_prbs, min_req_prbs]), high=np.array([max_prbs_inf, max_req_prbs, max_req_prbs]), dtype=int)

        # Creating function of req_prbs
        sine_time_range = np.arange(0, 10, 0.5)
        self.sine_amplitude = np.rint(max_req_prbs * np.sin(sine_time_range))
        self.time = 0
        self.state = np.array([13, self.sine_amplitude[self.time], self.sine_amplitude[self.time]])
        # remember: it's required prbs, not the amount that's there

    def step(self, action):
        # save the current state
        prev_state = tuple(self.state)
        # update the state
        prbs_inf = self.state[0] - action[0] - action[1]
        self.time = self.time + 1  # update time
        if self.time > 19:
            self.time = 0 # reset time
        self.state = np.array([prbs_inf, self.sine_amplitude[self.time], self.sine_amplitude[self.time]])

        # calculate reward
        # if slice gets the num prbs it requires, then reward
        if action[0] == prev_state[1] and action[1] == prev_state[2]:
            reward = 1
        else:
            reward = -1

        if self.state[0] < 0:
            reward = -10
            self.state[0] = 0
        elif self.state[0] > max_prbs_inf:
            reward = -10
            self.state[0] = max_prbs_inf


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
q_values = np.zeros([np.size(prbs_inf_states)*np.size(prbs_req_s1_states)*np.size(prbs_req_s2_states),
                     np.size(prbs_req_s1_states)*np.size(prbs_req_s2_states)])
row_indices = pd.MultiIndex.from_product([prbs_inf_states, prbs_req_s1_states, prbs_req_s2_states])
action1_space = np.arange(min_req_prbs, max_req_prbs+1)
action2_space = np.arange(min_req_prbs, max_req_prbs+1)
col_indices = pd.MultiIndex.from_product([action1_space, action2_space])
q_table = pd.DataFrame(q_values, columns=col_indices, index=row_indices)

# Hyperparameters
alpha = 0.7
gamma = 0.001
epsilon = 0.9999

# Training
for i in range(1, 10000):
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
env.state = np.array([15, env.sine_amplitude[env.time], env.sine_amplitude[env.time]])
print("Avail pRBs \t Req pRBs \t Alloc pRBs S1 \t Alloc pRBs S2")
for i in range(0, 20): # upper bound is exclusive
    # select action (exploit)
    action = q_table.loc[state].idxmax()  # Exploit learned values
    print("{} \t \t {} \t \t {}   \t \t \t {}".format(env.state[0], env.sine_amplitude[env.time], action[0], action[1]))
    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)
    # update state
    state = next_state


