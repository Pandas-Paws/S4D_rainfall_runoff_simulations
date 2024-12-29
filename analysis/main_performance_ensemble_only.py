import pickle
import sys
import numpy as np
import pandas as pd
import os
from performance_functions import (baseflow_index, bias, flow_duration_curve,
                                   get_quant, high_flows, low_flows, nse, alpha_nse, beta_nse,
                                   kge, stdev_rat, zero_freq, FHV, FLV, mass_balance)


# File name of ensemble dictionary is a user input
experiment = sys.argv[1]
note = sys.argv[2]
epoch_num = sys.argv[3]

# Load ensemble file
fname = f"results_data/{experiment}_{note}_{epoch_num}.pkl" # todo {experiment}_{note}.pkl
with open(fname, 'rb') as f:
    ens_dict = pickle.load(f)

# Directory to save CSV files
output_dir = "stats/ssm_metrics/"
os.makedirs(output_dir, exist_ok=True)

# Directory to save time series data
timeseries_output_dir = os.path.join(output_dir, f"time_series_{note}_{epoch_num}/")
os.makedirs(timeseries_output_dir, exist_ok=True)

# Dictionary to store performance metrics for each seed
seed_stats = {}

# Calculate performance measures for ensembles
for basin in ens_dict:
    # Extract all seeds
    seeds = [col.split('_')[-1] for col in ens_dict[basin].columns if col.startswith("qsim_")]

    for seed in seeds:

        # Filter dataframe for the current seed
        df_seed = ens_dict[basin].filter(regex=f"qsim_{seed}").join(ens_dict[basin]["qobs"])

        # Save the time series for the current seed
        timeseries_filename = f"{timeseries_output_dir}{basin}_timeseries_seed_{seed}.csv"
        df_seed.to_csv(timeseries_filename)
        print(f"Saved time series to {timeseries_filename}")


        # Calculate performance measures for the current seed
        obs5, sim5 = get_quant(df_seed, 0.05, seed)
        obs95, sim95 = get_quant(df_seed, 0.95, seed)
        obs0, sim0 = zero_freq(df_seed, seed)
        obsH, simH = high_flows(df_seed, seed)
        obsL, simL = low_flows(df_seed, seed)
        e_fhv_2 = FHV(df_seed, 2, seed)
        e_fhv_5 = FHV(df_seed, 5, seed)
        e_fhv_10 = FHV(df_seed, 10, seed)
        e_flv = FLV(df_seed, seed, 0.3)
        e_nse = nse(df_seed, seed)
        e_nse_alpha = alpha_nse(df_seed, seed)
        e_nse_beta = beta_nse(df_seed, seed)
        e_kge, r, alpha, beta = kge(df_seed, seed)
        massbias_total, massbias_pos, massbias_neg = mass_balance(df_seed, seed)
        e_bias = bias(df_seed, seed)
        e_stdev_rat = stdev_rat(df_seed, seed)
        obsFDC, simFDC = flow_duration_curve(df_seed, seed)

        # Store metrics in the dictionary
        if seed not in seed_stats:
            seed_stats[seed] = []

        stats = {
            'basin': basin,
            'nse': e_nse,
            'alpha_nse': e_nse_alpha,
            'beta_nse': e_nse_beta, 
            'kge': e_kge,
            'kge_r': r,
            'kge_alpha': alpha,
            'kge_beta': beta,  
            'fhv_2': e_fhv_2,
            'fhv_5': e_fhv_5,
            'fhv_10': e_fhv_10,
            'flv': e_flv, 
            'massbias_total': massbias_total,
            'massbias_pos': massbias_pos,
            'massbias_neg': massbias_neg, 
            'bias': e_bias,
            'stdev': e_stdev_rat,
            'obs5': obs5,
            'sim5': sim5,
            'obs95': obs95,
            'sim95': sim95,
            'obs0': obs0,
            'sim0': sim0,
            'obsL': obsL,
            'simL': simL,
            'obsH': obsH,
            'simH': simH,
            'obsFDC': obsFDC,
            'simFDC': simFDC
        }
        
        seed_stats[seed].append(stats)

# Save metrics for each seed to a separate CSV file
for seed, metrics in seed_stats.items():
    stats_df = pd.DataFrame(metrics,
                            columns=[
                                'basin', 'nse', 'alpha_nse', 'beta_nse', 'kge', 'kge_r', 'kge_alpha', 'kge_beta',
                                'fhv_2', 'fhv_5', 'fhv_10', 'flv', 'massbias_total', 'massbias_pos', 'massbias_neg',
                                'bias', 'stdev', 'obs5', 'sim5', 'obs95', 'sim95', 'obs0',
                                'sim0', 'obsL', 'simL', 'obsH', 'simH', 'obsFDC', 'simFDC'
                            ])

    # Calculate mean and median, excluding NaN values
    mean_stats = stats_df.mean(skipna=True)
    median_stats = stats_df.median(skipna=True)

    # Add mean and median rows to the DataFrame
    mean_stats['basin'] = 'mean'
    median_stats['basin'] = 'median'
    stats_df = stats_df.append(mean_stats, ignore_index=True)
    stats_df = stats_df.append(median_stats, ignore_index=True)

    # Save metrics for this seed to a CSV file
    csv_fname = f"{output_dir}{experiment}_{seed}_{note}_{epoch_num}.csv"
    stats_df.to_csv(csv_fname, index=False)
    print(f"Saved {csv_fname}")

print("All performance metrics saved.")