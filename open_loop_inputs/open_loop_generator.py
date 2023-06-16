import csv

def generate_open_loop_commands(file_path, constant_force_values, steering_angle_range, steering_angle_increment, Data_size):
    # Open the CSV file for writing
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(['Fx_fl', 'Fx_fr', 'Fx_rl', 'Fx_rr', 'steering_angle'])

        # Write the data rows
        steering_angle = 0
        row = constant_force_values + [steering_angle]
        writer.writerow(row)

        direction = 'right'
        for i in range(Data_size):
            if direction == 'right'and steering_angle <= steering_angle_range:
                steering_angle += steering_angle_increment
            elif direction == 'right'and steering_angle > steering_angle_range:
                direction = 'left'
                steering_angle -= steering_angle_increment
            if direction == 'left'and steering_angle >= (-steering_angle_range):
                steering_angle -= steering_angle_increment
            elif direction == 'left'and (steering_angle) < (-steering_angle_range):
                direction = 'right'
                steering_angle += steering_angle_increment

            row = constant_force_values + [steering_angle]
            writer.writerow(row)
            

# Example usage
Data_size = 1000
file_path = 'open_loop_commands.csv'
constant_force_values = [300, 300, 300, 300]  # Example constant force values
steering_angle_range = 0.3  # Oscillating steering angle range: -30 to 30, increment of 5
steering_angle_increment = 0.01  # Increment value for steering angle

generate_open_loop_commands(file_path, constant_force_values, steering_angle_range, steering_angle_increment, Data_size)
