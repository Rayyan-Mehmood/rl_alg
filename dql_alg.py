import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot
import tensorflow as tf
import keras
from keras.models import load_model
from collections import deque

num_slices = 3
prbs_inf_init = 8
max_prbs_inf = 14
max_req_prbs = 2  # max total prbs a slice can require
min_req_prbs = 0
max_curr_prbs = max_req_prbs
min_curr_prbs = 0
max_allocate = max_req_prbs
min_allocate = -max_req_prbs


class RicEnv:
    def __init__(self):
        # Encoding the action space as a Multi-Index array of tuples
        action1_space = np.arange(min_allocate, max_allocate + 1)
        action2_space = np.arange(min_allocate, max_allocate + 1)
        action3_space = np.arange(min_allocate, max_allocate + 1)
        self.action_space_MI = pd.MultiIndex.from_product([action1_space, action2_space, action3_space])
        self.action_space = pd.DataFrame(index=self.action_space_MI, columns=['Value'])

        # state = [prbs_inf, req_prbs_s1, req_prbs_s2, curr_prbs_s1, curr_prbs_s2]
        # self.prbs_req_data = np.genfromtxt('data_sin.csv', delimiter=',') # loading from a csv file
        # Creating function of req_prbs
        sine_time_range = np.arange(0, 10, 0.5) # generating manually
        self.prbs_req_data = np.rint((max_req_prbs / 2) * np.sin(sine_time_range) + max_req_prbs / 2)  # already changed to cosine
        self.time = 0
        self.episode_length = 20
        self.state = np.array([prbs_inf_init, self.prbs_req_data[self.time], self.prbs_req_data[self.time],
                               self.prbs_req_data[self.time], min_curr_prbs, min_curr_prbs, min_curr_prbs])

    # Calculates the individual reward for each slice based on the slice requirements and the amount of pRBs it
    # currently has
    def calc_individual_reward(self, r, a):
        # r = number of pRBs slice requires
        # a = number of pRBs slice currently has
        if a == r:
            individual_reward = 120
        elif a > r:
            individual_reward = 100 - (10 * (a - r))
        else:
            individual_reward = -100 - (10 * (r - abs(a)))

        return individual_reward

    def step(self, action):
        # save the current state
        prev_state = list(self.state)
        prbs_inf = prev_state[0]
        req_prbs_s1 = prev_state[1]
        req_prbs_s2 = prev_state[2]
        req_prbs_s3 = prev_state[3]
        curr_prbs_s1 = prev_state[4]
        curr_prbs_s2 = prev_state[5]
        curr_prbs_s3 = prev_state[6]
        # update the state
        new_prbs_inf = self.state[0] - action[0] - action[1] - action[2]
        new_curr_prbs_s1 = curr_prbs_s1 + action[0]
        new_curr_prbs_s2 = curr_prbs_s2 + action[1]
        new_curr_prbs_s3 = curr_prbs_s3 + action[2]
        self.time = self.time + 1  # update time
        if self.time >= self.episode_length:
            self.time = 0  # reset time
        self.state = np.array([new_prbs_inf, self.prbs_req_data[self.time], self.prbs_req_data[self.time],
                               self.prbs_req_data[self.time], new_curr_prbs_s1, new_curr_prbs_s2, new_curr_prbs_s3])

        # Calculate Reward

        # If not enough pRBs available, calculate the ideal action and then update the required pRBs for each slice
        # for example, if there are 2 pRBs available and each slice has 0 pRBs and requires 2 pRBs,
        # the ideal action would be to allocate 1 to each slice. Thus, the required pRBs for each slice will become 1.
        if prbs_inf - (req_prbs_s1 - curr_prbs_s1) - (req_prbs_s2 - curr_prbs_s2) - (req_prbs_s3 - curr_prbs_s3) < 0:
            difference_s1 = req_prbs_s1 - curr_prbs_s1
            difference_s2 = req_prbs_s2 - curr_prbs_s2
            difference_s3 = req_prbs_s3 - curr_prbs_s3
            # calculating the optimal actions
            if difference_s1 < 0 and difference_s2 > 0 and difference_s3 > 0:
                prbs_inf = prbs_inf - difference_s1
                oa1 = difference_s1
                oa2 = round((difference_s2 / (difference_s2 + difference_s3 - 0.00001)) * prbs_inf)
                oa3 = round((difference_s3 / (difference_s2 + difference_s3 + 0.00001)) * prbs_inf)
            elif difference_s1 > 0 and difference_s2 < 0 and difference_s3 > 0:
                prbs_inf = prbs_inf - difference_s2
                oa2 = difference_s2
                oa1 = round((difference_s1 / (difference_s1 + difference_s3 - 0.00001)) * prbs_inf)
                oa3 = round((difference_s3 / (difference_s1 + difference_s3 + 0.00001)) * prbs_inf)
            elif difference_s1 > 0 and difference_s2 > 0 and difference_s3 < 0:
                prbs_inf = prbs_inf - difference_s3
                oa3 = difference_s3
                oa1 = round((difference_s1 / (difference_s1 + difference_s2 - 0.00001)) * prbs_inf)
                oa2 = round((difference_s2 / (difference_s1 + difference_s2 + 0.00001)) * prbs_inf)
            elif difference_s1 < 0 and difference_s2 < 0 and difference_s3 > 0:
                prbs_inf = prbs_inf - difference_s1 - difference_s2
                oa1 = difference_s1
                oa2 = difference_s2
                oa3 = prbs_inf
            elif difference_s1 < 0 and difference_s2 > 0 and difference_s3 < 0:
                prbs_inf = prbs_inf - difference_s1 - difference_s3
                oa1 = difference_s1
                oa2 = prbs_inf
                oa3 = difference_s3
            elif difference_s1 > 0 and difference_s2 < 0 and difference_s3 < 0:
                prbs_inf = prbs_inf - difference_s2 - difference_s3
                oa1 = prbs_inf
                oa2 = difference_s2
                oa3 = difference_s3
            else:
                # all differences are positive
                oa1 = round((difference_s1 / (difference_s1 + difference_s2 + difference_s3)) * prbs_inf)
                oa2 = round((difference_s2 / (difference_s1 + difference_s2 + difference_s3)) * prbs_inf)
                oa3 = prbs_inf - oa1 - oa2  # allocate whatever is left

            prev_state[1] = round(prev_state[4] + oa1)
            prev_state[2] = round(prev_state[5] + oa2)
            prev_state[3] = round(prev_state[6] + oa3)

        # Calculate the individual reward for each slice and then sum the rewards
        total_reward = 0
        individual_reward = 0
        for slice in range(1, num_slices + 1):
            individual_reward = self.calc_individual_reward(prev_state[slice], self.state[slice + num_slices])
            total_reward = total_reward + individual_reward
        reward = total_reward

        # Impossible actions (which lead to prbs_inf being negative for e.g.) are punished heavily
        if new_prbs_inf < 0:
            reward = -150
            self.state[0] = 0  # set prbs_inf to zero
        elif new_prbs_inf > max_prbs_inf:
            self.state[0] = max_prbs_inf

        if new_curr_prbs_s1 < 0:
            reward = -150
            self.state[4] = 0
        elif new_curr_prbs_s1 > max_curr_prbs:
            self.state[4] = max_curr_prbs

        if new_curr_prbs_s2 < 0:
            reward = -150
            self.state[5] = 0
        elif new_curr_prbs_s2 > max_curr_prbs:
            self.state[5] = max_curr_prbs

        if new_curr_prbs_s3 < 0:
            reward = -150
            self.state[6] = 0
        elif new_curr_prbs_s3 > max_curr_prbs:
            self.state[6] = max_curr_prbs

        if self.time == 0:
            self.state[0] = prbs_inf_init
            self.state[3] = min_curr_prbs
            self.state[4] = min_curr_prbs

        info = {}
        done = False
        truncated = False
        return tuple(self.state), reward, done, truncated, info


class RicAgent:
    def build_net(self, state_shape, num_actions):
        """ The agent maps X-states to Y-actions
        e.g. The neural network output is [.1, .7, .1, .3]
        The highest value 0.7 is the Q-Value.
        The index of the highest action (0.7) is action #1.
        """
        learning_rate = 0.01
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=state_shape))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(num_actions, activation='linear'))
        model.compile(optimizer=optimizer, loss='mse')
        # model.summary()
        return model

    def train(self, replay_memory, model, done):
        MIN_REPLAY_SIZE = 500
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64
        mini_batch = random.sample(replay_memory, batch_size)

        # Collect observations from the mini-batch
        X = np.array([transition[0] for transition in mini_batch])

        # Predict Q-values for the entire batch
        current_qs_list = model.predict(X)

        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            current_qs = current_qs_list[index]
            current_qs[action] = reward

            Y.append(current_qs)

        model.fit(X, np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def test_other_inputs(env, main_model):
    # Testing other inputs
    print("TESTING OTHER INPUTS")
    env.state = np.array([8, 2, 2, 2, 0, 0, 0])
    print("State: ", env.state)
    predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten()
    action = env.action_space_MI[np.argmax(predictions)]
    print("Action: ")
    print(action)

    env.state = np.array([4, 0, 0, 0, 2, 2, 2])
    print("State: ", env.state)
    predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten()
    action = env.action_space_MI[np.argmax(predictions)]
    print("Action: ")
    print(action)


def test(env, main_model):
    # Testing
    # initializing the environment
    env.time = 0
    env.state = np.array([prbs_inf_init, env.prbs_req_data[env.time], env.prbs_req_data[env.time],
                               env.prbs_req_data[env.time], min_curr_prbs, min_curr_prbs, min_curr_prbs])

    print("pRBs_inf \t req_s1 \t req_s2 \t req_s3 \t pRBs_s1 \t pRBs_s2 \t pRBs_s3 \t A1 \t A2 \t A3")
    for i in range(0, env.episode_length):  # upper bound is exclusive
        # select action (exploit)
        predicted = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten()  # Exploit learned values
        # print(np.reshape(np.round(predicted, 3), (5, 5)))
        action = env.action_space_MI[np.argmax(predicted)]
        print("{} \t \t \t {} \t \t {} \t \t {} \t \t {} \t \t {} \t \t {} \t \t {}  \t {}  \t {}"
              .format(env.state[0], env.state[1], env.state[2], env.state[3], env.state[4],
                      env.state[5], env.state[6], action[0], action[1], action[2]))
        # move 1 step forward
        next_state, reward, done, truncated, info = env.step(action)

    # Test other inputs and print out the results
    # test_other_inputs(env, main_model)

def main():

    env = RicEnv()
    state = tuple(env.state)
    agent = RicAgent()

    # main_model = agent.build_net(env.state.shape, len(env.action_space))
    main_model = load_model("test_23_3.h5")
    replay_memory = deque(maxlen=50_000)

    # Hyper-parameters
    epsilon = 0.7  # want to explore all the time

    # Training
    for i in range(1, 5000):
        print(i)
        # select action (explore vs exploit)
        # if random.uniform(0, 1) < epsilon:
        #     action = env.action_space.sample().index[0]  # Explore action space
        #     action_index = env.action_space_MI.get_loc(action)
        # else:
        #     predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten() # Exploit learned values
        #     action = env.action_space_MI[np.argmax(predictions)]
        #     action_index = env.action_space_MI.get_loc(action)

        if random.uniform(0, 1) < 0.5:
            env.state = np.array([8, 2, 2, 2, 0, 0, 0])
            state = tuple(env.state)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample().index[0]  # Explore action space
                action_index = env.action_space_MI.get_loc(action)
            else:
                predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten() # Exploit learned values
                action = env.action_space_MI[np.argmax(predictions)]
                action_index = env.action_space_MI.get_loc(action)
        else:
            env.state = np.array([4, 0, 0, 0, 2, 2, 2])
            state = tuple(env.state)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample().index[0]  # Explore action space
                action_index = env.action_space_MI.get_loc(action)
            else:
                predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten() # Exploit learned values
                action = env.action_space_MI[np.argmax(predictions)]
                action_index = env.action_space_MI.get_loc(action)

        # move 1 step forward
        next_state, reward, done, truncated, info = env.step(action)
        replay_memory.append([state, action_index, reward, next_state, done])
        # Train main network
        if i % 10 == 0:
            agent.train(replay_memory, main_model, done)
        # update state
        state = next_state

    # Save model
    main_model.save("test_23_3.h5")
    # Testing
    # test(env, main_model)
    test_other_inputs(env, main_model)


if __name__ == "__main__":
    main()

