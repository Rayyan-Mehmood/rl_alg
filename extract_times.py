import re
import matplotlib.pyplot as plt
import os

# Define the file name
file_name = 't32_3_logfile.txt'

# Read the log file
with open(file_name, 'r') as file:
    log_text = file.read()

# Extract target cumulative rewards and corresponding times
target_rewards = re.findall(r'Time taken for (\d+) Reward: (\d+\.\d+)', log_text)

# Convert target rewards and times to lists
target_rewards, time_taken_seconds = zip(*[(int(reward), float(time)) for reward, time in target_rewards])

# Convert time taken from seconds to minutes
time_taken_minutes = [time / 60 for time in time_taken_seconds]

# Plot time versus target cumulative reward
plt.plot(target_rewards, time_taken_minutes, marker='o')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Cumulative Reward', fontsize=18)
plt.ylabel('Time Taken (minutes)', fontsize=18)
plt.title('Time Taken vs Cumulative Reward', fontsize=18)
plt.grid(True)
plt.tight_layout()
base_name = os.path.splitext(file_name)[0]
base_name = base_name.replace('_logfile', '_time')
plt.savefig(f'{base_name}.jpg')
plt.show()
plt.clf()