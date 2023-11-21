import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete


class RicEnv(Env):
    def __init__(self):
        # a0=allocate -2, a1=allocate -1, a2=nothing, a3=allocate 1, a4=allocate 2

        self.action_space = Box(low=np.array([-2, -2]), high=np.array([2, 2]), dtype=int)
        self.observation_space = Box(low=np.array([0, -2, -2]), high=np.array([10, 2, 2]), dtype=int)

        # self.action_space = MultiDiscrete(np.array([4, 4]), dtype=int)
        # self.observation_space = MultiDiscrete([10, 4, 4])
        self.state = np.array([10, 2, 2]) # remember: it's required prbs, not the amount that's there

    def step(self, action):
        prev_state = self.state
        # prbs_req_s1 = self.state[1] - action[0]  # current prbs required by slice1 =prev required prbs-allocated prbs
        # prbs_req_s2 = self.state[2] - action[1]
        prbs_inf = self.state[0] - (action[0] + action[1])
        # generate new required prbs randomly
        if prbs_inf < 5:
            prbs_req_s1 = random.randint(-2,0)
            prbs_req_s2 = random.randint(-2, 0)
        else:
            prbs_req_s1 = random.randint(0, 2)
            prbs_req_s2 = random.randint(0, 2)

        self.state = np.array([prbs_inf, prbs_req_s1, prbs_req_s2])

        # if prbs_inf is negative, then penalise
        if self.state[0] < 0:
            reward = -1
        # else if slice gets the num prbs it requires, then reward
        elif (action[0] == prev_state[1]) and (action[1] == prev_state[2]):
            reward = 1
        else:
            reward = 0

        info = {}
        done = False
        truncated = False
        return self.state, reward, done, truncated, info


env = RicEnv()
# print("Action Space {}".format(env.observation_space))
print(env.observation_space.sample())
print(env.action_space.sample())
# print(env.state)
# next_state, reward, done, truncated, info = env.step(np.array([2, 2]))
# state = next_state
# print(next_state)
# print(reward)
# print(env.observation_space.shape)
# x = np.array([[0], [0], [0]])
# print(x.size)
# q_table = np.zeros([env.observation_space.shape[0], env.action_space.shape[0]])
# num possible states = 10 * 5 * 5
# num possible actions = 5 * 5

q_table = np.zeros([250, 25])
# print(q_table.shape)
# print(q_table.size)

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.99

for i in range(1, 1000):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[env.state])  # Exploit learned values
    print(action)
    next_state, reward, done, truncated, info = env.step(action)
    old_value = q_table[env.state, action] # perhaps need to use state or prev_state instead of env.state
    next_max = np.max(q_table[next_state])
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[env.state, action] = new_value
    state = next_state


print(q_table)
