import numpy as np
import gymnasium as gym
import random
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import pandas as pd
import matplotlib.pyplot as plot
import tensorflow as tf
import keras
from collections import deque

prbs_inf_init = 8
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

        self.prbs_req_data = np.genfromtxt('data_sin.csv', delimiter=',')
        self.time = 0
        self.state = np.array([prbs_inf_init, self.prbs_req_data[self.time], self.prbs_req_data[self.time], min_curr_prbs, min_curr_prbs])
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
        self.state = np.array([new_prbs_inf, self.prbs_req_data[self.time], self.prbs_req_data[self.time], new_curr_prbs_s1, new_curr_prbs_s2])

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

    def agent(self, state_shape, action_shape):
        """ The agent maps X-states to Y-actions
        e.g. The neural network output is [.1, .7, .1, .3]
        The highest value 0.7 is the Q-Value.
        The index of the highest action (0.7) is action #1.
        """
        learning_rate = 0.001
        init = keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])
        # model.summary()
        return model

    def train(self, replay_memory, model, target_model, done):
        learning_rate = 0.999
        discount_factor = 0.00001

        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64 * 2
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)
        model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)



# create environment
env = RicEnv()
state = tuple(env.state)

# Encoding the action space as a Multi-Index array of tuples
action1_space = np.arange(min_allocate, max_allocate+1)
action2_space = np.arange(min_allocate, max_allocate+1)
action_space_MI = pd.MultiIndex.from_product([action1_space, action2_space])

# 1. Initialize the Target and Main models
# Main model
main_model = env.agent(env.observation_space.shape, 25)
# print(env.observation_space.shape)
# print(main_model.predict(env.state.reshape([1, env.state.shape[0]])))
# print(env.state.reshape([1, env.state.shape[0]]))
# Target Model (updated every 100 steps)
target_model = env.agent(env.observation_space.shape, 25)
target_model.set_weights(main_model.get_weights())

replay_memory = deque(maxlen=50_000)

# Hyperparameters
alpha = 0.99999
gamma = 0.00001
epsilon = 0.9999

# Training
for i in range(1, 20000):
    print(i)
    # select action (explore vs exploit)
    if random.uniform(0, 1) < epsilon:
        action = tuple(env.action_space.sample())  # Explore action space
        action_index = action_space_MI.get_loc(action)
    else:
        predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten() # Exploit learned values
        action = action_space_MI[np.argmax(predictions)]
        action_index = action_space_MI.get_loc(action)

    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)
    replay_memory.append([state, action_index, reward, next_state, done])
    # Train main network
    if i % 4 == 0:
        env.train(replay_memory, main_model, target_model, done)
    # update state
    state = next_state

    if i % 50 == 0:
        target_model.set_weights(main_model.get_weights())

# Testing
# initializing the environment
env.time = 0
env.state = np.array([prbs_inf_init, env.prbs_req_data[env.time], env.prbs_req_data[env.time], min_curr_prbs, min_curr_prbs])
state = tuple(env.state)
print("pRBs_inf \t pRBs_req \t pRBs_s1 \t pRBs_s2 \t A1 \t A2")
# prbs_s1_record = np.zeros(20)
# prbs_s2_record = np.zeros(20)
for i in range(0, 20): # upper bound is exclusive
    # prbs_s1_record[env.time] = env.state[3]
    # prbs_s2_record[env.time] = env.state[4]
    # select action (exploit)
    predicted = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten()  # Exploit learned values
    print(predicted)
    action = action_space_MI[np.argmax(predicted)]
    print("{} \t \t {} \t \t {} \t \t {} \t \t {}  \t {}".format(env.state[0],
                                                                             env.prbs_req_data[env.time], env.state[3],
                                                                             env.state[4],
                                                                             action[0], action[1]))
    # move 1 step forward
    next_state, reward, done, truncated, info = env.step(action)

    # update state
    state = next_state

