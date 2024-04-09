import matplotlib.pyplot as plt
import os

# Define the file name
file_name = 't32_3_logfile.txt'

# Function to extract data from a file
def extract_data(file_name):
    episode_numbers = []
    cumulative_rewards = []

    with open(file_name, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i % 10 == 0:  # Extract every 10th instance
                if line.startswith('Episode:'):
                    episode_number = int(line.split(':')[1].strip())
                    episode_numbers.append(episode_number)
                elif line.startswith('Cumulative Reward:'):
                    cumulative_reward = int(line.split(':')[1].strip())
                    cumulative_rewards.append(cumulative_reward)

    return episode_numbers, cumulative_rewards

# Extract data from the file
episode_numbers, cumulative_rewards = extract_data(file_name)

# Find the maximum length among the arrays
max_length = max(len(episode_numbers), len(cumulative_rewards))

# Pad arrays with None values if lengths are different
episode_numbers += [None] * (max_length - len(episode_numbers))
cumulative_rewards += [None] * (max_length - len(cumulative_rewards))

# Plot reward versus episode curve
plt.plot(episode_numbers, cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title(os.path.basename(file_name))

# Show plot
plt.show()