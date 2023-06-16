import rosbag
import csv

def rosbag_to_csv(rosbag_file, output_csv):
    # Open the ROSbag file
    bag = rosbag.Bag(rosbag_file, "r")
    # Create a CSV file for writing
    with open(output_csv, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        #Write the headers
        column_headers = ['Fx_fl', 'Fx_fr', 'Fx_rl', 'Fx_rr', 'steering_angle']
        writer.writerow(column_headers)

        # Iterate over each message in the ROSbag
        for topic, msg, t in bag.read_messages():
            # Extract the necessary data from the message and store it
            if topic == '/control/car_command':
                writer.writerow([msg.Fx_fl[0], msg.Fx_fr[0], msg.Fx_rl[0], msg.Fx_rr[0], msg.steering_angle[0]])
    # Close the ROSbag file
    bag.close()

# Usage example
rosbag_file = 'fsg.bag'
output_csv = 'fsg.csv'

rosbag_to_csv(rosbag_file, output_csv)
