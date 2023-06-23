import pandas as pd
df1 = pd.read_csv('car_command_for_sim_PLEASE.csv')

# Read the second CSV file
df2 = pd.read_csv('state_for_sim_PLEASE.csv')

# Concatenate the columns horizontally
combined_df = pd.concat([df1, df2], axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('open_loop_sim_amzsim_model_PLEASE.csv', index=False)