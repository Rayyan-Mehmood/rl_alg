import matplotlib.pyplot as plot
import os

# Define the file name
file_name = 'ql_5x5_logfile.txt'

# Function to extract data from a file
def extract_data(file_name):
    episode_numbers = []
    cumulative_rewards = []

    with open(file_name, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:  # Extract every 10th instance
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

# Convert episode numbers to thousands
episode_numbers = [episode / 1000 if episode is not None else None for episode in episode_numbers]


# Plot reward versus episode curve
plot.plot(episode_numbers, cumulative_rewards)
plot.title('Reward Curve', fontsize=18)
plot.grid(True)
plot.xticks(fontsize=16)
plot.yticks(fontsize=16)
plot.xlabel("Episode ($\\times 10^3$)", fontsize=18)
plot.ylabel("Cumulative Reward", fontsize=18)
plot.tight_layout()
base_name = os.path.splitext(file_name)[0]
base_name = base_name.replace('_logfile', '_reward')
plot.savefig(f'{base_name}.jpg')
plot.show()
plot.clf()