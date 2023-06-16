import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def objective_function(x):
    return x[0]**2 + x[1]**2

num_particles = 50
max_iterations = 100
lower_bounds = [-10, -10]  # Lower bounds for x and y
upper_bounds = [10, 10]    # Upper bounds for x and y

particle_positions = np.random.uniform(lower_bounds, upper_bounds, (num_particles, 2))
particle_velocities = np.zeros((num_particles, 2))
best_particle_positions = particle_positions.copy()
global_best_position = best_particle_positions[np.argmin([objective_function(p) for p in particle_positions])]
particle_positions_history = []

for _ in range(max_iterations):
    particle_positions_history.append(particle_positions.copy())

    for i in range(num_particles):
        inertia = 0.5
        cognitive_rate = 0.5
        social_rate = 0.5
        r1, r2 = np.random.rand(2)
        cognitive_component = cognitive_rate * r1 * (best_particle_positions[i] - particle_positions[i])
        social_component = social_rate * r2 * (global_best_position - particle_positions[i])
        particle_velocities[i] = inertia * particle_velocities[i] + cognitive_component + social_component

        particle_positions[i] += particle_velocities[i]
        particle_positions[i] = np.clip(particle_positions[i], lower_bounds, upper_bounds)

        if objective_function(particle_positions[i]) < objective_function(best_particle_positions[i]):
            best_particle_positions[i] = particle_positions[i]

        if objective_function(best_particle_positions[i]) < objective_function(global_best_position):
            global_best_position = best_particle_positions[i]

print("Optimal solution:")
print("x =", global_best_position[0])
print("y =", global_best_position[1])
print("Minimum value =", objective_function(global_best_position))

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

def update_plot(frame):
    plt.cla()
    plt.plot(*zip(*particle_positions_history[frame]), 'go', markersize=6)  # Plot particle positions
    plt.contourf(X, Y, Z, levels=np.linspace(0, 100, 50), cmap='viridis')  # Plot contour
    plt.xlabel('x')
    plt.ylabel('y')

fig = plt.figure()
ani = FuncAnimation(fig, update_plot, frames=len(particle_positions_history), interval=200)


plt.show()
