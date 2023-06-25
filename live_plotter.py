import numpy as np
import matplotlib.pyplot as plt
from IPython import display

data = np.genfromtxt('./Data/Data_four_wheel/all_thetas_dirac.csv', delimiter=',', names=True, dtype=None)
T_peak = data['T_peak']
T_slope = data['T_slope']
accepted = data['accepted']

# Separate the columns

# Create a color map based on column3 values
colors = ['red' if value == 0 else 'blue' for value in accepted]

# Initialize the plot
plt.figure()
legend_elements = [
    plt.Line2D([0], [0], marker='X', color='w', label='Start', markerfacecolor='orange', markersize=8),
    plt.Line2D([0], [0], marker='X', color='w', label='Goal', markerfacecolor='green', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Accepted', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Rejected', markerfacecolor='red', markersize=8)
]
cross_size = 50
plt.legend(handles=legend_elements, loc='upper left')
plt.scatter(0.37, 0.4, color='green', marker='x', s=cross_size)
plt.scatter(0.5, 0.5, color='orange', marker='x', s=cross_size)
plt.pause(10)

# Iterate over the data points
for i in range(len(T_peak) // 2):
    # Clear the previous plot
    plt.clf()

    # Plot the points up to the current iteration
    for j in range(i):
        plt.scatter(0.37, 0.4, color='green', marker='x', s=cross_size)
        plt.scatter(0.5, 0.5, color='orange', marker='x', s=cross_size)
        plt.scatter(T_peak[j], T_slope[j], color=colors[j], marker='o')

    # Set plot properties
    plt.xlabel(r'$ \theta_{T_{peak}} \rm [\;] $')
    plt.ylabel(r'$ \theta_{T_{slope}} \rm [\;]$')
    #plt.title(f'Iteration {i+1}')

    # Display the fixed legend
    legend_elements = [
        plt.Line2D([0], [0], marker='X', color='w', label='Start', markerfacecolor='orange', markersize=8),
        plt.Line2D([0], [0], marker='X', color='w', label='Goal', markerfacecolor='green', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Accepted', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Rejected', markerfacecolor='red', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # Display the updated plot
    display.clear_output(wait=True)
    display.display(plt.gcf())

    # Pause to create the animation effect
    plt.pause(0.1)

# Clear the final plot
plt.clf()

# Set plot properties
plt.xlabel('T_peak \rm [\;]')
plt.ylabel('T_slope \rm [\;]')
plt.title('Final Result')

# Plot all the points at once with fixed legend
for i in range(len(T_peak)):
    plt.scatter(T_peak[i], T_slope[i], color=colors[i], marker='o')

# Display the fixed legend
plt.legend(['Rejected', 'Accepted'], loc='upper right')

# Display the final plot
plt.show()
