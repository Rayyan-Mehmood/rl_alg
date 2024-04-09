import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot

num_slices = 3
max_req_prbs = 3  # max total prbs a slice can require
min_req_prbs = 1
max_curr_prbs = 3
min_curr_prbs = 1
max_allocate = max_req_prbs - min_curr_prbs
min_allocate = -max_allocate
max_data_element = 22
prbs_inf_init = round(50 / ((1/max_req_prbs)*max_data_element))
max_prbs_inf = prbs_inf_init
num_episodes = 100
episode_length = 10
testing_iterations = 20
test_frequency = 50
show_plots = True

def convert_data_range(data):

    num_bins = max_req_prbs + 1
    # Calculate the thresholds for each bin
    thresholds = np.linspace(-0.00001, max_data_element, num_bins)
    thresholds = np.append(thresholds, np.inf)
    # Initialize an array to store the converted data
    converted_data = np.zeros_like(data, dtype=int)
    # Assign values to each bin
    for i in range(num_bins - 1):
        converted_data[(data > thresholds[i]) & (data <= thresholds[i + 1])] = i + 1
    converted_data[(data > thresholds[max_req_prbs])] = max_req_prbs


    return converted_data

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
        excel_file_path = 'data/data_real_10_slices.xlsx'
        data_frame = pd.read_excel(excel_file_path, skiprows=120, header=None)
        prbs_req_data_s1 = data_frame[2].values  # row C
        prbs_req_data_s2 = data_frame[9].values  # row J
        prbs_req_data_s3 = data_frame[6].values  # row G
        # self.prbs_req_s1 = np.round(np.array(prbs_req_data_s1)).astype(int)
        # self.prbs_req_s2 = np.round(np.array(prbs_req_data_s2)).astype(int)
        # self.prbs_req_s3 = np.round(np.array(prbs_req_data_s3)).astype(int)
        self.prbs_req_s1 = convert_data_range(prbs_req_data_s1)
        self.prbs_req_s2 = convert_data_range(prbs_req_data_s2)
        self.prbs_req_s3 = convert_data_range(prbs_req_data_s3)
        self.init_state()

        self.q_table = pd.DataFrame(columns=self.action_space_MI, dtype=np.float64)

    def init_state(self):
        self.time = 0
        self.state = np.array([prbs_inf_init, random.randint(min_req_prbs, max_req_prbs), random.randint(min_req_prbs, max_req_prbs),
                               random.randint(min_req_prbs, max_req_prbs), min_curr_prbs, min_curr_prbs, min_curr_prbs])

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
        # update the state (required prbs)
        if not testing:
            # during training, we generate the required prbs data randomly
            self.state = np.array([new_prbs_inf, random.randint(min_req_prbs, max_req_prbs), random.randint(min_req_prbs, max_req_prbs),
                                   random.randint(min_req_prbs, max_req_prbs), new_curr_prbs_s1, new_curr_prbs_s2,
                                   new_curr_prbs_s3])
        else:
            # during testing, we take the required prbs from the real data
            self.state = np.array([new_prbs_inf, self.prbs_req_s1[self.time], self.prbs_req_s2[self.time],
                                   self.prbs_req_s3[self.time], new_curr_prbs_s1, new_curr_prbs_s2,
                                   new_curr_prbs_s3])

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

        # Bounding the states...
        if bounded:
            if new_prbs_inf < 0:
                reward = -1000
                self.state[0] = 0  # set prbs_inf to zero
            elif new_prbs_inf > max_prbs_inf:
                self.state[0] = max_prbs_inf

            if new_curr_prbs_s1 < min_curr_prbs:
                reward = -1000
                self.state[4] = min_curr_prbs
            elif new_curr_prbs_s1 > max_curr_prbs:
                self.state[4] = max_curr_prbs

            if new_curr_prbs_s2 < min_curr_prbs:
                reward = -1000
                self.state[5] = min_curr_prbs
            elif new_curr_prbs_s2 > max_curr_prbs:
                self.state[5] = max_curr_prbs

            if new_curr_prbs_s3 < min_curr_prbs:
                reward = -1000
                self.state[6] = min_curr_prbs
            elif new_curr_prbs_s3 > max_curr_prbs:
                self.state[6] = max_curr_prbs


        info = {}
        if self.time >= episode_length:
            done = True
        else:
            done = False
        truncated = False
        return tuple(self.state), reward, done, truncated, info

def test(env):
    # load q-table
    # env.q_table = pd.read_pickle('q_table_train.pkl')
    # initializing the environment
    env.time = 0
    env.state = np.array([prbs_inf_init, env.prbs_req_s1[env.time], env.prbs_req_s2[env.time],
                          env.prbs_req_s3[env.time], min_curr_prbs, min_curr_prbs, min_curr_prbs])

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
        print(state)
        # select action (exploit)
        env.check_state_exist(str(state))
        action = env.q_table.loc[str(state)].idxmax()  # Exploit learned values
        q_value = env.q_table.loc[str(state)].max()
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
            if prev_allocated + action[slice - 1] < 0:  # impossible action
                num_impossible_actions += 1
                num_under_allocations += 1
                total_under_allocated += prev_req - (prev_allocated + action[slice - 1])
            elif allocated < prev_req:  # under-allocation
                num_under_allocations += 1
                total_under_allocated += prev_req - allocated
            elif allocated > prev_req:  # over-allocation
                num_over_allocations += 1
                total_over_allocated += allocated - prev_req
            else:
                num_correct_allocations += 1
        new_prbs_inf = state[0] - action[0] - action[1] - action[2]
        if new_prbs_inf < 0:
            num_impossible_actions += 1

        state = next_state

    # total_under_allocated = total_under_allocated * ((1/max_req_prbs)*max_data_element)
    # total_over_allocated = total_over_allocated * ((1 / max_req_prbs) * max_data_element)

    with open('logfile.txt', 'a') as f:
        f.write("Number of Impossible Actions: " + str(num_impossible_actions) + "\n")
        f.write("Number of Under-allocations: " + str(num_under_allocations) + "\n")
        f.write("Number of Over-allocations: " + str(num_over_allocations) + "\n")
        f.write("Number of Correct Allocations: " + str(num_correct_allocations) + "\n")
        f.write("Total amount Under-allocated: " + str(total_under_allocated) + "\n")
        f.write("Total amount Over-allocated: " + str(total_over_allocated) + "\n")
        f.write("Cumulative Reward: " + str(cumulative_reward) + "\n\n")

    print("Number of Impossible Actions: ", num_impossible_actions)
    print("Number of Under-allocations: ", num_under_allocations)
    print("Number of Over-allocations: ", num_over_allocations)
    print("Number of Correct Allocations: ", num_correct_allocations)
    print("Total amount Under-allocated: ", total_under_allocated)
    print("Total amount Over-allocated: ", total_over_allocated)
    print("Cumulative Reward: ", cumulative_reward)

    if show_plots:
        min_y_axis = int(min(min(env.prbs_req_s1[:testing_iterations]), min(s1_record),
                             min(env.prbs_req_s2[:testing_iterations]), min(s2_record),
                             min(env.prbs_req_s3[:testing_iterations]), min(s3_record)))
        max_y_axis = int(max(max(env.prbs_req_s1[:testing_iterations]), max(s1_record),
                             max(env.prbs_req_s2[:testing_iterations]), max(s2_record),
                             max(env.prbs_req_s3[:testing_iterations]), max(s3_record)))

        plot.plot(np.arange(0, testing_iterations), env.prbs_req_s1[:testing_iterations], marker='o', label='Required')
        plot.plot(np.arange(0, testing_iterations), s1_record, marker='x', label='Allocated')
        plot.title('Slice 1', fontsize=18)
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 2), fontsize=16)
        # plot.ylim(min_y_axis-0.1, max_y_axis+0.1)
        plot.yticks(np.arange(int(min(min(env.prbs_req_s1[:testing_iterations]), min(s1_record))),
                              int(max(max(env.prbs_req_s1[:testing_iterations]), max(s1_record))) + 1, 1))
        plot.ylim(int(min(min(env.prbs_req_s1[:testing_iterations]), min(s1_record))) - 0.1,
                  int(max(max(env.prbs_req_s1[:testing_iterations]), max(s1_record))) + 0.1)
        plot.yticks(fontsize=16)
        plot.xlabel("Timestep", fontsize=18)
        plot.ylabel("Number of pRBs", fontsize=18)
        plot.legend(fontsize=16)
        plot.tight_layout()
        plot.savefig('Slice_1.jpg')
        plot.show()
        plot.clf()

        plot.plot(np.arange(0, testing_iterations), env.prbs_req_s2[:testing_iterations], marker='o', label='Required')
        plot.plot(np.arange(0, testing_iterations), s2_record, marker='x', label='Allocated')
        plot.title('Slice 2', fontsize=18)
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 2), fontsize=16)
        # plot.ylim(min_y_axis - 0.1, max_y_axis + 0.1)
        plot.yticks(np.arange(int(min(min(env.prbs_req_s2[:testing_iterations]), min(s2_record))),
                              int(max(max(env.prbs_req_s2[:testing_iterations]), max(s2_record))) + 1, 1))
        plot.ylim(int(min(min(env.prbs_req_s2[:testing_iterations]), min(s2_record))) - 0.1,
                  int(max(max(env.prbs_req_s2[:testing_iterations]), max(s2_record))) + 0.1)
        plot.yticks(fontsize=16)
        plot.xlabel("Timestep", fontsize=18)
        plot.ylabel("Number of pRBs", fontsize=18)
        plot.legend(fontsize=16)
        plot.tight_layout()
        plot.savefig('Slice_2.jpg')
        plot.show()
        plot.clf()

        plot.plot(np.arange(0, testing_iterations), env.prbs_req_s3[:testing_iterations], marker='o', label='Required')
        plot.plot(np.arange(0, testing_iterations), s3_record, marker='x', label='Allocated')
        plot.title('Slice 3', fontsize=18)
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 2), fontsize=16)
        # plot.ylim(min_y_axis - 0.1, max_y_axis + 0.1)
        plot.yticks(np.arange(int(min(min(env.prbs_req_s3[:testing_iterations]), min(s3_record))),
                              int(max(max(env.prbs_req_s3[:testing_iterations]), max(s3_record))) + 1, 1))
        plot.ylim(int(min(min(env.prbs_req_s3[:testing_iterations]), min(s3_record))) - 0.1,
                  int(max(max(env.prbs_req_s3[:testing_iterations]), max(s3_record))) + 0.1)
        plot.yticks(fontsize=16)
        plot.xlabel("Timestep", fontsize=18)
        plot.ylabel("Number of pRBs", fontsize=18)
        plot.legend(fontsize=16)
        plot.tight_layout()
        plot.savefig('Slice_3.jpg')
        plot.show()
        plot.clf()

        plot.plot(np.arange(0, testing_iterations), s1_record, marker='x', label='Slice 1')
        plot.plot(np.arange(0, testing_iterations), s2_record, marker='s', label='Slice 2')
        plot.plot(np.arange(0, testing_iterations), s3_record, marker='p', label='Slice 3')
        plot.title('All Slices', fontsize=18)
        plot.grid(True)
        plot.xticks(np.arange(0, testing_iterations, 2), fontsize=16)
        plot.yticks(np.arange(min_y_axis, max_y_axis + 1, 1))
        plot.ylim(min_y_axis - 0.1, max_y_axis + 0.1)
        plot.yticks(fontsize=16)
        plot.xlabel("Timestep", fontsize=18)
        plot.ylabel("Number of pRBs Allocated", fontsize=18)
        plot.legend(fontsize=16)
        plot.tight_layout()
        plot.savefig('Slice_All.jpg')
        plot.show()
        plot.clf()

    # env.q_table.to_csv('q_table_test.csv')
    # env.q_table.to_pickle('q_table_test.pkl')
    return cumulative_reward

def main():
    # create environment
    env = RicEnv()
    state = tuple(env.state)

    # Hyperparameters
    alpha = 0.99999
    gamma = 0.00001
    epsilon = 0.9999

    cumulative_rewards = []
    with open('logfile_ql.txt', 'w') as f:
        f.write(".... " + "\n")

    # Training
    for episode in range(1, num_episodes+1):
        env.init_state()
        state = tuple(env.state)
        for i in range(1, episode_length+1):
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
            # updating the q-table
            env.q_table.loc[str(state), action] = new_value
            # update state
            state = next_state

        if episode % test_frequency == 0:
            with open('logfile.txt', 'a') as f:
                f.write("Episode: " + str(episode) + "\n")
            # Plot reward
            cumulative_reward = test(env)
            cumulative_rewards.append(cumulative_reward)
            plot.plot(np.arange(test_frequency, episode + test_frequency, test_frequency), cumulative_rewards, marker='o')
            plot.title('Cumulative Reward')
            plot.grid(True)
            plot.savefig('Reward.jpg')
            plot.show()
            plot.clf()
            # Save model
            # env.q_table.to_csv('Test_23_train.csv')
            # env.q_table.to_pickle('Test_23_train.pkl')

    # Testing
    # test(env)


if __name__ == "__main__":
    main()
