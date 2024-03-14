import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot
import tensorflow as tf
import keras
from keras.models import load_model
from collections import deque


# Parameters
num_slices = 3
prbs_inf_init = 3
max_prbs_inf = prbs_inf_init
max_req_prbs = 1  # max total prbs a slice can require
min_req_prbs = 0
max_curr_prbs = 1
min_curr_prbs = 0
max_allocate = 1
min_allocate = -1
num_episodes = 100
episode_length = 10
testing_iterations = 10
initial_epsilon = 0.99
final_epsilon = 0.01
impossible_action_reward = -5
show_plots = True
# names of the files from which to load the models and save the models to
old_main_model = "test_29_1_m.h5"
new_main_model = "test_29_1_m.h5"
old_target_model = "test_29_1_t.h5"
new_target_model = "test_29_1_t.h5"


class RicEnv:
    def __init__(self):
        # Encoding the action space as a Multi-Index array of tuples
        action1_space = np.arange(min_allocate, max_allocate + 1)
        action2_space = np.arange(min_allocate, max_allocate + 1)
        action3_space = np.arange(min_allocate, max_allocate + 1)
        self.action_space_MI = pd.MultiIndex.from_product([action1_space, action2_space, action3_space])
        self.action_space = pd.DataFrame(index=self.action_space_MI, columns=['Value'])

        # using the required prbs data from the real data
        # excel_file_path = 'data/data_real_10_slices.xlsx'
        # data_frame = pd.read_excel(excel_file_path, skiprows=1, header=None)
        # prbs_req_data_s1 = data_frame[2].values  # row C
        # prbs_req_data_s2 = data_frame[3].values  # row D
        # prbs_req_data_s3 = data_frame[6].values  # row D
        # self.prbs_req_s1 = np.round(np.array(prbs_req_data_s1)).astype(int)
        # self.prbs_req_s2 = np.round(np.array(prbs_req_data_s2)).astype(int)
        # self.prbs_req_s3 = np.round(np.array(prbs_req_data_s3)).astype(int)
        # generating required prbs from a cosine wave
        # sine_time_range = np.arange(0, 10, 0.5) # generating manually
        # self.prbs_req_data = np.rint((max_req_prbs / 2) * np.sin(sine_time_range) + max_req_prbs / 2)  # already changed to cosine
        # initializing the state
        self.time = 0
        self.state = np.array([prbs_inf_init, random.randint(0, max_req_prbs), random.randint(0, max_req_prbs),
                               random.randint(0, max_req_prbs), min_curr_prbs, min_curr_prbs, min_curr_prbs])

    def init_state(self):
        self.time = 0
        self.state = np.array([prbs_inf_init, random.randint(0, max_req_prbs), random.randint(0, max_req_prbs),
                               random.randint(0, max_req_prbs), min_curr_prbs, min_curr_prbs, min_curr_prbs])


    # testing is true when we are testing the model
    # bounded is true when we want to bound the state space during training so that the agent never explores
    # out-of-bounds states during training
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
        # update the state (inf_prbs and curr_prbs)
        new_prbs_inf = self.state[0] - action[0] - action[1] - action[2]
        new_curr_prbs_s1 = curr_prbs_s1 + action[0]
        new_curr_prbs_s2 = curr_prbs_s2 + action[1]
        new_curr_prbs_s3 = curr_prbs_s3 + action[2]
        # update the time
        self.time = self.time + 1
        # update the state (required prbs)
        if not testing:
            # during training, we generate the required prbs data randomly
            self.state = np.array([new_prbs_inf, random.randint(0, max_req_prbs), random.randint(0, max_req_prbs),
                               random.randint(0, max_req_prbs), new_curr_prbs_s1, new_curr_prbs_s2, new_curr_prbs_s3])
        else:
            # during testing, we take the required prbs from the real data
            self.state = np.array([new_prbs_inf, self.prbs_req_s1[self.time], self.prbs_req_s2[self.time],
                               self.prbs_req_s3[self.time], new_curr_prbs_s1, new_curr_prbs_s2,
                                   new_curr_prbs_s3])

        # Calculate Reward

        # This if branch updates the required prbs when there are not enough prbs available.
        # For example, if each slice requires 2 prbs but there are only 3 prbs available, the required prbs will be
        # updated to 1 (because the best possible action is to allocate 1 prb to each slice). Therefore, it will be
        # possible for the allocated to be equal to the required and for the agent to receive max reward
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

        # Calculate the reward for each slice
        # e.g. if any of the slices are under allocated, 'under' will be flagged as true and total reward will be -5
        impossible = False
        under = False
        over = False
        for slice in range(1, num_slices + 1):
            a = self.state[slice + num_slices]
            r = prev_state[slice]
            if a < 0:
                impossible = True
            elif a < r:
                under = True
            elif a > r:
                over = True

        if new_prbs_inf < 0:
            impossible = True

        if impossible:
            reward = -5
        elif under:
            reward = -5
        elif over:
            reward = 5
        else:
            reward = 10

        # Bounding the states...
        if bounded:
            if new_prbs_inf < 0:
                self.state[0] = 0  # set prbs_inf to zero
            elif new_prbs_inf > max_prbs_inf:
                self.state[0] = max_prbs_inf

            if new_curr_prbs_s1 < 0:
                self.state[4] = 0
            elif new_curr_prbs_s1 > max_curr_prbs:
                self.state[4] = max_curr_prbs

            if new_curr_prbs_s2 < 0:
                self.state[5] = 0
            elif new_curr_prbs_s2 > max_curr_prbs:
                self.state[5] = max_curr_prbs

            if new_curr_prbs_s3 < 0:
                self.state[6] = 0
            elif new_curr_prbs_s3 > max_curr_prbs:
                self.state[6] = max_curr_prbs


        info = {}
        if self.time >= episode_length:
            done = True
        else:
            done = False
        truncated = False
        return tuple(self.state), reward, done, truncated, info

class RicAgent:
    def build_net(self, state_shape, num_actions):
        learning_rate = 0.01
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=state_shape))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # model.summary()
        return model

    def train(self, replay_memory, model, target_model):
        learning_rate = 0.7
        discount_factor = 0.3

        MIN_REPLAY_SIZE = 500
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64
        mini_batch = random.sample(replay_memory, batch_size)

        # Collect observations from the mini-batch
        current_states = np.array([transition[0] for transition in mini_batch])
        # Predict Q-values for the entire batch
        current_qs_list = model.predict(current_states)
        next_states = np.array([transition[3] for transition in mini_batch])
        next_qs_list = target_model.predict(next_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            max_new_q = reward + discount_factor * np.max(next_qs_list[index])
            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_new_q

            X.append(observation)
            Y.append(current_qs)

        model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


# Testing with hardcoded inputs
def test_other_inputs(env, main_model, target_model):
    def print_testcase(env, main_model, state, target_model):
        env.state = state
        print("State: ", env.state)
        predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten()
        action = env.action_space_MI[np.argmax(predictions)]
        q_value = np.max(predictions)
        print("Action: ", action)
        print("Main Model Q-value: ", q_value)

    # Testing other inputs
    print("TESTING OTHER INPUTS")

    print_testcase(env, main_model, np.array([3, 0, 0, 0, 0, 0, 0]), target_model)
    print_testcase(env, main_model, np.array([3, 1, 1, 1, 0, 0, 0]), target_model)
    print_testcase(env, main_model, np.array([2, 0, 0, 0, 1, 0, 0]), target_model)
    print_testcase(env, main_model, np.array([2, 1, 1, 0, 0, 1, 0]), target_model)
    print_testcase(env, main_model, np.array([2, 0, 0, 1, 0, 1, 0]), target_model)
    print_testcase(env, main_model, np.array([1, 1, 0, 0, 1, 0, 1]), target_model)
    print_testcase(env, main_model, np.array([1, 1, 1, 1, 1, 0, 1]), target_model)
    print_testcase(env, main_model, np.array([1, 0, 0, 0, 0, 1, 1]), target_model)
    print_testcase(env, main_model, np.array([0, 1, 1, 1, 1, 1, 1]), target_model)
    print_testcase(env, main_model, np.array([0, 0, 0, 0, 1, 1, 1]), target_model)


# Testing with the real data
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
        next_state, reward, done, truncated, info = env.step(action, True, True)

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

    main_model = agent.build_net(env.state.shape, len(env.action_space)) # build the model
    # main_model = load_model(old_main_model) # load the model

    target_model = agent.build_net(env.state.shape, len(env.action_space))
    # target_model = load_model(old_target_model)
    target_model.set_weights(main_model.get_weights())

    replay_memory = deque(maxlen=50_000)
    epsilon = initial_epsilon

    # Training
    for episode in range(1, num_episodes + 1):

        env.init_state()
        state = tuple(env.state)

        for i in range(1, episode_length+1):
            print("Episode: ", episode, "Timestep: ", i)
            # select action (explore vs exploit)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample().index[0]
                action_index = env.action_space_MI.get_loc(action)
            else:
                predictions = main_model.predict(env.state.reshape([1, env.state.shape[0]])).flatten() # Exploit learned values
                action = env.action_space_MI[np.argmax(predictions)]
                action_index = env.action_space_MI.get_loc(action)

            # move 1 step forward
            next_state, reward, done, truncated, info = env.step(action, False, False)
            replay_memory.append([state, action_index, reward, next_state, done])
            # update state
            state = next_state

        # after each episode, update epsilon
        epsilon = initial_epsilon - (episode / num_episodes) * (initial_epsilon - final_epsilon)
        # Train main network every n episodes
        if episode % 10 == 0:
            agent.train(replay_memory, main_model, target_model)

        # Update target network weights every n episodes
        if episode % 30 == 0:
            target_model.set_weights(main_model.get_weights())



    # Save model
    main_model.save(new_main_model)
    target_model.save(new_target_model)
    # Testing
    # test(env, main_model)  # test on real data
    test_other_inputs(env, main_model, target_model)  # test on hardcoded inputs

if __name__ == "__main__":
    main()

