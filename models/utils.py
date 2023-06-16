import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import csv
import os

def circle_residuals(params, points):
    # Calculate residuals between circle and points
    cx, cy, r = params
    residuals = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r
    return residuals

def fit_circle(x_coords, y_coords):
    # Combine x and y coordinates into points array
    points = np.column_stack((x_coords, y_coords))

    # Initial guess for circle parameters (center and radius)
    initial_params = [0, 0, 1]

    # Minimize the residuals to find the best-fit circle
    result = least_squares(circle_residuals, initial_params, args=(points,), method='trf')

    # Extract the optimized circle parameters
    cx, cy, r = result.x

    return r


def RK45_integrate(f, t0, tf, x0, dt):
    t = [t0]
    x = [x0]
    
    torque = 1000
    inputs = [0.1, torque, torque, torque, torque]

    while t[-1] < tf:
        
        x_current = x[-1]
        t_current = t[-1]

        # Perform one step of RK4 integration
        k1 = f(x_current, inputs)
        k2 = f(x_current + 0.5 * dt * k1, inputs)
        k3 = f(x_current + 0.5 * dt * k2, inputs)
        k4 = f(x_current + dt * k3, inputs)
        x_next = x_current + (1.0 / 6.0) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update time and state
        t.append(t_current + dt)
        x.append(x_next)

    return np.array(t), np.array(x)


def plot_column_histograms(data, legends):
    """
    Plots a histogram for each column of a 2D numpy array.
    """
    num_cols = data.shape[1]  # Get the number of columns in the data
    
    # Plot a histogram for each column
    for i in range(num_cols):
        plt.hist(data[:, i], bins=10)  # Plot the histogram
        
        # Set the plot title and axis labels
        plt.xlabel(legends[i])
        plt.ylabel("Frequency")
        
        plt.show()  # Show the plot

def get_csv_row_count(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        row_count = len(list(reader))
    return row_count


def get_folder_path():
    folder_path = os.path.abspath("")
    return folder_path



