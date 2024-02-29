import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot

num_slices = 3
max_prbs_inf = 12
prbs_inf_init = max_prbs_inf
max_req_prbs = 6  # max total prbs a slice can require
min_req_prbs = 0
max_curr_prbs = max_req_prbs
min_curr_prbs = 0
max_allocate = max_req_prbs
min_allocate = -max_req_prbs

class RicEnv():
    def __init__(self):
        # Here, the bounds are inclusive
        # action = [allocate_prbs_s1, allocate_prbs_s2]
        action1_space = np.arange(min_allocate, max_allocate + 1)
        action2_space = np.arange(min_allocate, max_allocate + 1)
        action3_space = np.arange(min_allocate, max_allocate + 1)
        self.action_space_MI = pd.MultiIndex.from_product([action1_space, action2_space, action3_space])
        self.action_space = pd.DataFrame(index=self.action_space_MI, columns=['Value'])

        # state = [prbs_inf, req_prbs_s1, req_prbs_s2, curr_prbs_s1, curr_prbs_s2]
        # self.episode_length = np.arange(0, 10, 0.5)
        # self.prbs_req_data = np.rint((max_req_prbs / 2) * np.sin(episode_length) + max_req_prbs / 2)
        # self.prbs_req_data = np.genfromtxt('data_sin.csv', delimiter=',')
        self.episode_length = 1  # length of data is 1897
        excel_file_path = 'data/data_real_10_slices.xlsx'
        data_frame = pd.read_excel(excel_file_path, skiprows=1, header=None)
        prbs_req_data_s1 = data_frame[2].values  # row C
        prbs_req_data_s2 = data_frame[3].values  # row D
        prbs_req_data_s3 = data_frame[6].values  # row D
        self.prbs_req_s1 = np.round(np.array(prbs_req_data_s1)).astype(int)
        self.prbs_req_s2 = np.round(np.array(prbs_req_data_s2)).astype(int)
        self.prbs_req_s3 = np.round(np.array(prbs_req_data_s3)).astype(int)
        self.init_state()

        self.q_table = pd.DataFrame(columns=self.action_space_MI, dtype=np.float64)

    def init_state(self):
        self.time = 0
        self.state = np.array([prbs_inf_init, self.prbs_req_s1[self.time], self.prbs_req_s2[self.time],
                               self.prbs_req_s3[self.time], min_curr_prbs, min_curr_prbs, min_curr_prbs])

    def calc_individual_reward(self, r, a):
        if a == r:
            individual_reward = 120
        elif a > r:
            individual_reward = 100 - (10 * (a - r))
        else:
            individual_reward = -100 + (10 * (r - abs(a)))

        return individual_reward

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.action_space), index=self.q_table.columns, name=state)
            self.q_table.loc[state] = new_row

    def step_test(self, action):
        # update the state
        new_prbs_inf = self.state[0] - action[0] - action[1] - action[2]
        new_curr_prbs_s1 = self.state[4] + action[0]
        new_curr_prbs_s2 = self.state[5] + action[1]
        new_curr_prbs_s3 = self.state[6] + action[2]
        self.time = self.time + 1  # update time
        if self.time > self.episode_length:
            self.time = 0 # reset time
        self.state = np.array([new_prbs_inf, self.prbs_req_s1[self.time], self.prbs_req_s2[self.time],
                               self.prbs_req_s3[self.time], new_curr_prbs_s1, new_curr_prbs_s2, new_curr_prbs_s3])

        info = {}
        done = False
        truncated = False
        reward = 0
        return tuple(self.state), reward, done, truncated, info

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
        if self.time > self.episode_length:
            self.time = 0 # reset time
        self.state = np.array([new_prbs_inf, self.prbs_req_s1[self.time], self.prbs_req_s2[self.time],
                               self.prbs_req_s3[self.time], new_curr_prbs_s1, new_curr_prbs_s2, new_curr_prbs_s3])

        # calculate reward
        # if not enough pRBs available
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

        total_reward = 0
        individual_reward = 0
        for slice in range(1, num_slices+1):
            individual_reward = self.calc_individual_reward(prev_state[slice], self.state[slice+num_slices])
            total_reward = total_reward + individual_reward
        reward = total_reward

        if new_prbs_inf < 0:
            reward = -1000
            self.state[0] = 0
        elif new_prbs_inf > max_prbs_inf:
            self.state[0] = max_prbs_inf

        if new_curr_prbs_s1 < 0:
            reward = -1000
            self.state[4] = 0
        elif new_curr_prbs_s1 > max_curr_prbs:
            self.state[4] = max_curr_prbs

        if new_curr_prbs_s2 < 0:
            reward = -1000
            self.state[5] = 0
        elif new_curr_prbs_s2 > max_curr_prbs:
            self.state[5] = max_curr_prbs

        if new_curr_prbs_s3 < 0:
            reward = -1000
            self.state[6] = 0
        elif new_curr_prbs_s3 > max_curr_prbs:
            self.state[6] = max_curr_prbs


        info = {}
        done = False
        truncated = False
        return tuple(self.state), reward, done, truncated, info

def test(env):
    # load q-table
    env.q_table = pd.read_pickle('q_table_train.pkl')
    # initializing the environment
    env.init_state()

    state = tuple(env.state)
    print("pRBs_inf \t req_s1 \t req_s2 \t req_s3 \t pRBs_s1 \t pRBs_s2 \t pRBs_s3 \t A1 \t A2 \t A3")
    # prbs_s1_record = np.zeros(20)
    # prbs_s2_record = np.zeros(20)
    for i in range(0, env.episode_length):  # upper bound is exclusive
        # prbs_s1_record[env.time] = env.state[3]
        # prbs_s2_record[env.time] = env.state[4]
        # select action (exploit)
        env.check_state_exist(str(state))
        action = env.q_table.loc[str(state)].idxmax()  # Exploit learned values
        print("{} \t \t \t {} \t \t \t {} \t \t \t {} \t \t \t {} \t \t \t {} \t \t \t {} \t \t \t {}  \t {}  \t {}"
              .format(env.state[0], env.state[1], env.state[2], env.state[3],env.state[4],
                      env.state[5], env.state[6], action[0], action[1], action[2]))
        # move 1 step forward
        next_state, reward, done, truncated, info = env.step(action)

        # update state
        state = next_state

    # plot.plot(np.arange(0, 20), env.prbs_req_data, marker='o')
    # plot.title('pRBs Required')
    # plot.show()
    # plot.plot(np.arange(0, 20), prbs_s1_record, marker='o')
    # plot.title('pRBs Slice 1')
    # plot.show()
    # plot.plot(np.arange(0, 20), prbs_s2_record, marker='o')
    # plot.title('pRBs Slice 2')
    # plot.show()

    env.q_table.to_csv('q_table_test.csv')
    env.q_table.to_pickle('q_table_test.pkl')

def main():
    # create environment
    env = RicEnv()
    state = tuple(env.state)

    # Hyperparameters
    alpha = 0.99999
    gamma = 0.00001
    epsilon = 0.9999
    num_episodes = 1

    # Training
    for episode in range(1, num_episodes+1):
        env.init_state()
        state = tuple(env.state)
        for i in range(1, env.episode_length+1):
            print("Episode: ", episode, "Timestep: ", i)
            # select action (explore vs exploit)
            env.check_state_exist(str(state))
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample().index[0] # Explore action space
            else:
                action = env.q_table.loc[str(state)].idxmax()  # Exploit learned values
            # move 1 step forward
            next_state, reward, done, truncated, info = env.step(action)
            env.check_state_exist(str(next_state))
            # update q-value
            old_value = env.q_table.loc[str(state), action]
            next_max = env.q_table.loc[str(next_state)].max()
            new_value = (1 - alpha) * old_value + alpha * (reward + (gamma * next_max))
            # new_value = reward # not using bellman equation
            env.q_table.loc[str(state), action] = new_value
            # update state
            state = next_state

    # env.q_table.to_csv('Test_22_6.csv')
    # env.q_table.to_pickle('Test_22_6.pkl')

    # Testing
    #test(env)


if __name__ == "__main__":
    main()
