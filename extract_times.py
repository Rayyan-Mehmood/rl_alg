import re
import matplotlib.pyplot as plt
import os

# Define the list of file names
file_names = ['t32_3_logfile.txt', 't37_3_logfile.txt', 't35_7_logfile.txt', 't39_2_logfile.txt']

# Initialize lists to store all extracted data
all_target_rewards = []
all_time_taken_minutes = []

# Iterate over each file
for file_name in file_names:
    # Read the log file
    with open(file_name, 'r') as file:
        log_text = file.read()

    # Extract target cumulative rewards and corresponding times
    target_rewards = re.findall(r'Time taken for (\d+) Reward: (\d+\.\d+)', log_text)

    # Convert target rewards and times to lists
    target_rewards, time_taken_seconds = zip(*[(int(reward), float(time)) for reward, time in target_rewards])

    # Convert time taken from seconds to minutes
    time_taken_minutes = [time / 60 for time in time_taken_seconds]

    # Append to the lists
    all_target_rewards.append(target_rewards)
    all_time_taken_minutes.append(time_taken_minutes)

# Plot all curves on the same graph
for i in range(len(file_names)):
    plt.plot(all_target_rewards[i], all_time_taken_minutes[i], marker='o', label=os.path.basename(file_names[i]))

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Cumulative Reward', fontsize=18)
plt.ylabel('Time Taken (minutes)', fontsize=18)
plt.title('Time Taken vs Cumulative Reward', fontsize=18)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
plt.clf()