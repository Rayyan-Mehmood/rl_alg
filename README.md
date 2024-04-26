The code can be found at: https://github.com/Rayyan-Mehmood/rl_alg

Running the QL Code
1.	Copy ‘rl_alg.py’ to a folder. 
2.	Copy the excel file containing the data into a subfolder called ‘data’. Place that subfolder in the folder created in step 1. 
3.	Open rl_alg.py in a code editor and make the following adjustments as necessary:
  •	If you want to discretize the data, set discretize to True (line 8). Then adjust the min and max req_prbs and curr_prbs to the desired range (lines 10-13). For example, if you want the discretized range to be of size 2, you would set min_req_prbs to 1, max_req_prbs to 3 and likewise for curr_prbs. 
  •	If you don’t want to discretize the data, set discretize to False. Then, set min_req_prbs and min_curr_prbs to zero. Then, set max_req_prbs and max_curr_prbs appropriately (e.g. to 50). 
  •	If you want to change the columns of data that are used for testing, change lines 16 and 58-60. 
  •	If you want to modify the graphs of ‘Required vs Allocated pRBs’, adjust the test function. 
  •	If you want to change the number of episodes, change num_episodes (line 22).
  •	If you want to change the frequency of testing, change test_frequency (line 25).
4.	Run rl_alg.py
  •	The graphs  of ‘Required vs Allocated pRBs’ for each slice are saved as ‘Slice_1.jpg’, ‘Slice_2.jpg’ and ‘Slice_3.jpg’.
  •	If the agent reached a cumulative reward of 1000, the model at that point in training will be saved as ‘1000_m.h’. Similarly for the other target rewards.

Running the DQL Code
1.	Copy ‘dql_alg.py’ to a folder. 
2.	Copy the excel file containing the data into a subfolder called ‘data’. Place that subfolder in the folder created in step 1. 
3.	Open dql_alg.py in a code editor and make the following adjustments as necessary:
  •	If you want to discretize the data, set discretize to True (line 13). Then adjust the min and max req_prbs and curr_prbs to the desired range (lines 15-18). For example, if you want the discretized range to be of size 2, you would set min_req_prbs to 1, max_req_prbs to 3 and likewise for curr_prbs. 
  •	If you don’t want to discretize the data, set discretize to False. Then, set min_req_prbs and min_curr_prbs to zero. Then, set max_req_prbs and max_curr_prbs appropriately (e.g. to 50). 
  •	If you want to change the columns of data that are used for testing, change lines 21 and 75-77. 
  •	If you want to modify the graphs of ‘Required vs Allocated pRBs’, adjust the test function. 
  •	If you want to change the number of episodes, change num_episodes (line 27).
  •	If you want to change the frequency of testing, change test_frequency (line 31).
  •	If you want to build the neural nets from scratch, set Build to True. 
  •	If you want to load an existing neural net model, set Build to False and adjust the file names as necessary (lines 39-42). 
    	The models suffixed by ‘_m’ refer to the main network models and the models suffixed by ‘_t’ refer to the target network models.  
4.	Run dql_alg.py
  •	The graphs  of ‘Required vs Allocated pRBs’ for each slice are saved as ‘Slice_1.jpg’, ‘Slice_2.jpg’ and ‘Slice_3.jpg’.
  •	If the agent reached a cumulative reward of 100, the model at that point in training will be saved as ‘100_m.h’. Similarly for the other target rewards.  

Generating the Reward Curve Graph (e.g. Fig. 5.9 in my thesis)
To generate the graph of the reward curve for each model, you need to run ‘extract_values.py’. This code extracts the necessary measurements from the logfiles generated during the training of the model (i.e. when running dql_alg.py) and then produces a graph. You can edit the name of the log file used (line 5) if necessary. 

Generating the ‘Reward vs Training Time’ Graph (e.g. Fig. 5.10 in my thesis)
To generate the graph of the ‘Reward vs Training Time’ for each model, you need to run ‘extract_times.py’. This code extracts the necessary measurements from the logfiles generated during the training of the model (i.e. when running dql_alg.py) and then produces a graph. You can edit the name of the log files used (line 6) if necessary. 
