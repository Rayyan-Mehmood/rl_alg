import numpy as np
from itertools import product
import pandas as pd

prbs_inf_states = np.arange(0, 51)
prbs_req_s1_states = np.arange(-10, 11)
prbs_req_s2_states = np.arange(-10, 11)
# all_states = list(product(prbs_inf_states, prbs_req_s1_states, prbs_req_s2_states))
q_values = np.zeros([22491, 25])
# multi_index = pd.MultiIndex.from_tuples(all_states)
row_indices = pd.MultiIndex.from_product([prbs_inf_states, prbs_req_s1_states, prbs_req_s2_states])
action1_space = np.arange(-2, 3)
action2_space = np.arange(-2, 3)
col_indices = pd.MultiIndex.from_product([action1_space, action2_space])
q_table = pd.DataFrame(q_values, columns=col_indices, index=row_indices)
q_table.loc[(0, -2, 2), (-2, -2)] = 5
# return action with max q value for a given state:
# q_table.loc[(0, -2, 2)].idxmax()
# return max q value for a given state
# q_table.loc[(0, -2, 2)].max()
q_table = pd.read_csv('file_name.csv')
# print(q_table.loc[tuple(np.array([0, -2, -2])), (-2, -2)])
q_table.to_csv('delete.csv')