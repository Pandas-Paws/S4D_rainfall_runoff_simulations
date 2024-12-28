import os
import pickle
import pandas as pd

# Directory containing the .pkl files
config = 0
directory = f'results_data/sacsma_standard_training_results/config_{config}/'

# Initialize an empty dictionary to hold the combined data
combined_dict = {}

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file) 
            # The loaded data is a list of 4.
            # data[0]: parameters of Sac-SMA
            # data[1]: time series from 1980-01-01 to 2014-12-31
            df_temp = pd.DataFrame(data[1], columns=['qobs'])
            print(df_temp)

            # Merge the loaded data with the combined dictionary
            combined_dict.update(data)

# Save the combined dictionary into a single .pkl file
output_file = f'results_data/sacsma_config{config}.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(combined_dict, file)

print(f"Combined .pkl file saved at {output_file}")