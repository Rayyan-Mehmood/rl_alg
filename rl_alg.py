import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import pandas as pd


class RicEnv(Env):
    def __init__(self):
        # Here, the bounds are inclusive
        self.action_space = Box(low=np.array([-10, -10]), high=np.array([10, 10]), dtype=int)
        self.observation_space = Box(low=np.array([0, -10, -10]), high=np.array([50, 10, 10]), dtype=int)

        self.state = np.array([5, 2, 2]) # remember: it's required prbs, not the amount that's there

    def step(self, action):
        prev_state = tuple(self.state)

        new_prbs_inf = self.state[0] - (action[0] + action[1])
        if new_prbs_inf > 10 or new_prbs_inf < 0:
            prbs_inf = self.state[0]
        else:
            prbs_inf = new_prbs_inf
        # generate new required prbs randomly
        if prbs_inf < 5:
            prbs_req_s1 = random.randint(-2,0)
            prbs_req_s2 = random.randint(-2, 0)
        else:
            prbs_req_s1 = random.randint(0, 2)
            prbs_req_s2 = random.randint(0, 2)

        self.state = np.array([prbs_inf, prbs_req_s1, prbs_req_s2])

        # else if slice gets the num prbs it requires, then reward
        if action[0] == prev_state[1] and action[1] == prev_state[2]:
            reward = 1
        else:
            reward = 0

        info = {}
        done = False
        truncated = False
        return tuple(self.state), reward, done, truncated, info


# create environment
env = RicEnv()
state = tuple(env.state)

# initialize q table
# Here, the bounds are exclusive
prbs_inf_states = np.arange(0, 51)
prbs_req_s1_states = np.arange(-10, 11)
prbs_req_s2_states = np.arange(-10, 11)
q_values = np.zeros([22491, 441])
row_indices = pd.MultiIndex.from_product([prbs_inf_states, prbs_req_s1_states, prbs_req_s2_states])
action1_space = np.arange(-10, 11)
action2_space = np.arange(-10, 11)
col_indices = pd.MultiIndex.from_product([action1_space, action2_space])
q_table = pd.DataFrame(q_values, columns=col_indices, index=row_indices)

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.7

# Training
for i in range(1, 1000):
    # select action (explore vs exploit)
    if random.uniform(0, 1) < epsilon:
        action = tuple(env.action_space.sample())  # Explore action space
    else:
        action = q_table.loc[state].idxmax()  # Exploit learned values

    # move 1 step forward
    state = tuple(env.state)
    next_state, reward, done, truncated, info = env.step(action)
    # update q-value
    # old_value = q_table.loc[state, action]
    # next_max = q_table.loc[next_state].max()
    # new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    new_value = reward
    q_table.loc[state, action] = new_value
    # update state
    state = next_state

q_table.to_csv('file_name.csv')
