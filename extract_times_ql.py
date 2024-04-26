import re
import matplotlib.pyplot as plot

# Define the file name
file_name = 'ql_5x5_logfile.txt'

# Read the log file
with open(file_name, 'r') as file:
    log_text = file.read()

episodes = re.findall(r'Episode: (\d+)', log_text)
times = re.findall(r'Time: (\d+\.\d+)', log_text)

# Convert time from seconds to minutes
episodes = [int(episode) for episode in episodes]
times = [float(time) / 60 for time in times]

# Plot episode vs time
plot.plot(episodes, times, marker='o')
plot.grid(True)
plot.xticks(fontsize=16)
plot.yticks(fontsize=16)
plot.xlabel('Episode', fontsize=18)
plot.ylabel('Training Time (minutes)', fontsize=18)
plot.title('Episode vs Training Time', fontsize=18)
plot.tight_layout()
plot.savefig('ql_5x5_time.jpg')
plot.show()
plot.clf()