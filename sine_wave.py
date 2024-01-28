import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Get x values of the sine wave
time = np.arange(0, 10, 0.5)
# Amplitude of the sine wave is sine of a variable like time
# Note: -0.0 == 0
amplitude = np.rint((2/2) * np.sin(time) + 2/2)
# use the tofile() method
# and use ',' as a separator
# as we have to generate a csv file
amplitude.tofile('data_sin.csv', sep = ',')
# Plot a sine wave using time and amplitude obtained for the sine wave
plot.plot(time, amplitude)
# Give a title for the sine wave plot
plot.title('Sine wave')
# Give x axis label for the sine wave plot
plot.xlabel('Time')
# Give y axis label for the sine wave plot
plot.ylabel('Amplitude = sin(time)')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()
# Display the sine wave
plot.show()