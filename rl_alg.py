import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete


class RicEnv(Env):
    def __init__(self):
        self.action_space = Box(low=np.array([-2, -2]), high=np.array([2, 2]), dtype=int)
        self.observation_space = Box(low=np.array([0, -2, -2]), high=np.array([10, 2, 2]), dtype=int)

        self.state = np.array([10, 2, 2]) # remember: it's required prbs, not the amount that's there

    def step(self, action):
        prev_state = self.state

        prbs_inf = self.state[0] - (action[0] + action[1])
        # generate new required prbs randomly
        if prbs_inf < 5:
            prbs_req_s1 = random.randint(-2,0)
            prbs_req_s2 = random.randint(-2, 0)
        else:
            prbs_req_s1 = random.randint(0, 2)
            prbs_req_s2 = random.randint(0, 2)

        self.state = np.array([prbs_inf, prbs_req_s1, prbs_req_s2])

        # else if slice gets the num prbs it requires, then reward
        if (action[0] == prev_state[1]) and (action[1] == prev_state[2]):
            reward = 1
        else:
            reward = 0

        info = {}
        done = False
        truncated = False
        return self.state, reward, done, truncated, info


# create environment
env = RicEnv()
# initialize q table
q_table = np.zeros([250, 25])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.99

# Training
for i in range(1, 1000):
    # select action (explore vs exploit)
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[env.state])  # Exploit learned values
    print(action)

    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)
    # update q-value
    old_value = q_table[env.state, action]
    next_max = np.max(q_table[next_state])
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[env.state, action] = new_value
    # update state
    state = next_state

print(q_table)
