import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot
import tensorflow as tf
import keras
from keras.models import load_model
from collections import deque

num_slices = 3
prbs_inf_init = 50
max_prbs_inf = prbs_inf_init
max_req_prbs = 10  # max total prbs a slice can require
min_req_prbs = 0
max_curr_prbs = 35
min_curr_prbs = 0
max_allocate = 35
min_allocate = -35
num_iterations = 5000
testing_iterations = 50
epsilon = 0.5
impossible_action_reward = -150
show_plots = True
old_model = "test_28_7.h5"
new_model = "test_28_7.h5"
# initial_epsilon = 0.9
# final_epsilon = 0.4

class RicEnv:
    def __init__(self):
        # Encoding the action space as a Multi-Index array of tuples
        action1_space = np.arange(min_allocate, max_allocate + 1)
        action2_space = np.arange(min_allocate, max_allocate + 1)
        action3_space = np.arange(min_allocate, max_allocate + 1)
        self.action_space_MI = pd.MultiIndex.from_product([action1_space, action2_space, action3_space])
        self.action_space = pd.DataFrame(index=self.action_space_MI, columns=['Value'])

        excel_file_path = 'data/data_real_10_slices.xlsx'
        data_frame = pd.read_excel(excel_file_path, skiprows=1, header=None)
        prbs_req_data_s1 = data_frame[2].values  # row C
        prbs_req_data_s2 = data_frame[3].values  # row D
        prbs_req_data_s3 = data_frame[6].values  # row D
        self.prbs_req_s1 = np.round(np.array(prbs_req_data_s1)).astype(int)
        self.prbs_req_s2 = np.round(np.array(prbs_req_data_s2)).astype(int)
        self.prbs_req_s3 = np.round(np.array(prbs_req_data_s3)).astype(int)
        # Creating function of req_prbs
        # sine_time_range = np.arange(0, 10, 0.5) # generating manually
        # self.prbs_req_data = np.rint((max_req_prbs / 2) * np.sin(sine_time_range) + max_req_prbs / 2)  # already changed to cosine
        self.time = 0
        self.episode_length = 20
        self.state = np.array([prbs_inf_init, random.randint(0, max_req_prbs), random.randint(0, max_req_prbs),
                               random.randint(0, max_req_prbs), min_curr_prbs, min_curr_prbs, min_curr_prbs])

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

    def step(self, action, testing=False, bounded=True):
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
        if self.time >= self.episode_length and not testing:
            self.time = 0  # reset time
        if not testing:
            self.state = np.array([new_prbs_inf, random.randint(0, max_req_prbs), random.randint(0, max_req_prbs),
                               random.randint(0, max_req_prbs), new_curr_prbs_s1, new_curr_prbs_s2, new_curr_prbs_s3])
        else:
            self.state = np.array([new_prbs_inf, self.prbs_req_s1[self.time], self.prbs_req_s2[self.time],
                               self.prbs_req_s3[self.time], new_curr_prbs_s1, new_curr_prbs_s2,
                                   new_curr_prbs_s3])

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
            reward = impossible_action_reward
            if bounded: self.state[0] = 0  # set prbs_inf to zero
        elif new_prbs_inf > max_prbs_inf:
            if bounded: self.state[0] = max_prbs_inf

        if new_curr_prbs_s1 < 0:
            reward = impossible_action_reward
            if bounded: self.state[4] = 0
        elif new_curr_prbs_s1 > max_curr_prbs:
            if bounded: self.state[4] = max_curr_prbs

        if new_curr_prbs_s2 < 0:
            reward = impossible_action_reward
            if bounded: self.state[5] = 0
        elif new_curr_prbs_s2 > max_curr_prbs:
            if bounded: self.state[5] = max_curr_prbs

        if new_curr_prbs_s3 < 0:
            reward = impossible_action_reward
            if bounded: self.state[6] = 0
        elif new_curr_prbs_s3 > max_curr_prbs:
            if bounded: self.state[6] = max_curr_prbs

        if self.time == 0:
            self.state[0] = prbs_inf_init
            self.state[4] = min_curr_prbs
            self.state[5] = min_curr_prbs
            self.state[6] = min_curr_prbs

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
    def print_testcase(env, main_model, state):
        env.state = state
        print("State: ", env.state)
        predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten()
        action = env.action_space_MI[np.argmax(predictions)]
        q_value = np.max(predictions)
        print("Action: ", action)
        print("Q-value: ", q_value)
        print(predictions[env.action_space_MI.get_loc((2, 2, 2))])

    # Testing other inputs
    print("TESTING OTHER INPUTS")

    print_testcase(env, main_model, np.array([50, 2, 2, 2, 0, 0, 0]))
    print_testcase(env, main_model, np.array([40, 0, 0, 0, 2, 6, 2]))
    print_testcase(env, main_model, np.array([48, 0, 9, 7, 0, 0, 2]))
    print_testcase(env, main_model, np.array([40, 1, 6, 2, 6, 2, 2]))
    print_testcase(env, main_model, np.array([35, 1, 6, 8, 5, 5, 5]))
    print_testcase(env, main_model, np.array([33, 4, 0, 6, 7, 4, 6]))
    print_testcase(env, main_model, np.array([31, 10, 3, 3, 8, 2, 9]))


    # prbs req > 10
    # print_testcase(env, main_model, np.array([37, 25, 14, 3, 5, 6, 2]))
    # print_testcase(env, main_model, np.array([45, 0, 29, 17, 2, 1, 2]))
    # print_testcase(env, main_model, np.array([41, 21, 16, 20, 5, 1, 3]))
    # print_testcase(env, main_model, np.array([11, 20, 13, 13, 18, 12, 9]))


def test(env, main_model):
    # Testing
    # initializing the environment
    env.time = 0
    env.state = np.array([prbs_inf_init, env.prbs_req_s1[env.time], env.prbs_req_s2[env.time],
                           env.prbs_req_s3[env.time], 0, 0, 0])

    num_impossible_actions = 0
    num_under_allocations = 0
    num_over_allocations = 0
    num_correct_allocations = 0
    total_under_allocated = 0
    total_over_allocated = 0
    cumulative_reward = 0
    s1_record = np.zeros(testing_iterations)
    s2_record = np.zeros(testing_iterations)
    s3_record = np.zeros(testing_iterations)

    # print("State \t \t Action \t \t Q-value")
    for i in range(0, testing_iterations):  # upper bound is exclusive
        state = tuple(env.state)
        # select action (exploit)
        predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]]), verbose=0).flatten()  # Exploit learned values
        action = env.action_space_MI[np.argmax(predictions)]
        q_value = np.max(predictions)
        print(i, ": State: ", env.state, "\t Action: ", action, "\t Q-value: ", q_value)
        # print(' '.join(map(str, env.state)), ' '.join(map(str, action)), q_value)
        # move 1 step forward
        next_state, reward, done, truncated, info = env.step(action, True, False)

        # Test metrics
        cumulative_reward += reward
        s1_record[i] = next_state[4]
        s2_record[i] = next_state[5]
        s3_record[i] = next_state[6]
        for slice in range(1, num_slices + 1):
            prev_req = state[slice]
            allocated = next_state[slice + num_slices]
            prev_allocated = state[slice + num_slices]
            if prev_allocated + action[slice-1] < 0:  # impossible action
                num_impossible_actions += 1
                num_under_allocations += 1
                total_under_allocated += prev_req - (prev_allocated + action[slice-1])
            elif allocated < prev_req:  # under-allocation
                num_under_allocations += 1
                total_under_allocated += prev_req - allocated
            elif allocated > prev_req:  # over-allocation
                num_over_allocations += 1
                total_over_allocated += allocated - prev_req
            else:
                num_correct_allocations += 1

        state = next_state

    print("Number of Impossible Actions: ", num_impossible_actions)
    print("Number of Under-allocations: ", num_under_allocations)
    print("Number of Over-allocations: ", num_over_allocations)
    print("Number of Correct Allocations: ", num_correct_allocations)
    print("Total amount Under-allocated: ", total_under_allocated)
    print("Total amount Over-allocated: ", total_over_allocated)
    print("Cumulative Reward: ", cumulative_reward)

    if show_plots:
        plot.plot(np.arange(0, testing_iterations), env.prbs_req_s1[:testing_iterations], marker='o', label='Required')
        plot.plot(np.arange(0, testing_iterations), s1_record, marker='x', label='Allocated')
        plot.title('Slice 1')
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 4))
        plot.yticks(np.arange(int(min(min(env.prbs_req_s1[:testing_iterations]), min(s1_record))),
                              int(max(max(env.prbs_req_s1[:testing_iterations]), max(s1_record))) + 1, 4))
        plot.legend()
        plot.savefig('Slice_1.jpg')
        plot.show()

        plot.plot(np.arange(0, testing_iterations), env.prbs_req_s2[:testing_iterations], marker='o', label='Required')
        plot.plot(np.arange(0, testing_iterations), s2_record, marker='x', label='Allocated')
        plot.title('Slice 2')
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 4))
        plot.yticks(np.arange(int(min(min(env.prbs_req_s2[:testing_iterations]), min(s2_record))),
                              int(max(max(env.prbs_req_s2[:testing_iterations]), max(s2_record))) + 1, 4))
        plot.legend()
        plot.savefig('Slice_2.jpg')
        plot.show()

        plot.plot(np.arange(0, testing_iterations), env.prbs_req_s3[:testing_iterations], marker='o', label='Required')
        plot.plot(np.arange(0, testing_iterations), s3_record, marker='x', label='Allocated')
        plot.title('Slice 3')
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 4))
        plot.yticks(np.arange(int(min(min(env.prbs_req_s3[:testing_iterations]), min(s3_record))),
                              int(max(max(env.prbs_req_s3[:testing_iterations]), max(s3_record))) + 1, 4))
        plot.legend()
        plot.savefig('Slice_3.jpg')
        plot.show()


def main():

    env = RicEnv()
    state = tuple(env.state)
    agent = RicAgent()

    # main_model = agent.build_net(env.state.shape, len(env.action_space))
    main_model = load_model(old_model)
    # learning_rate = 0.5
    # optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    # main_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    replay_memory = deque(maxlen=50_000)

    # Training
    for i in range(1, num_iterations):
        print(i)
        # epsilon = initial_epsilon - (i / num_iterations) * (initial_epsilon - final_epsilon)
        # select action (explore vs exploit)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample().index[0]
            # action = (random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            action_index = env.action_space_MI.get_loc(action)
        else:
            predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten() # Exploit learned values
            action = env.action_space_MI[np.argmax(predictions)]
            action_index = env.action_space_MI.get_loc(action)

        # move 1 step forward
        next_state, reward, done, truncated, info = env.step(action, False, True)
        replay_memory.append([state, action_index, reward, next_state, done])
        # Train main network
        if i % 10 == 0:
            agent.train(replay_memory, main_model, done)
        # update state
        state = next_state

    # Save model
    main_model.save(new_model)
    # Testing
    # test(env, main_model)
    test_other_inputs(env, main_model)


if __name__ == "__main__":
    main()

