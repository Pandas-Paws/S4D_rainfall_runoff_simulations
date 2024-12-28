import glob
import os
import pickle
import sys
import pandas as pd
from multiprocessing import Pool

nSeeds = 1
firstSeed = 200
seeds = [200, 201, 202, 203, 204, 205, 206, 207]

# number of GPUs available
nGPUs = 8

# user inputs
experiment = sys.argv[1]
note = sys.argv[2]
epoch_num = sys.argv[3]


def run_evaluation(seed_gpu):
    seed, gpu = seed_gpu
    # get the correct run directory by reading the screen report
    fname = f"reports/{experiment}.{seed}_{note}.out"
    print(f"Working on seed: {seed} -- file: {fname}")
    
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if "Sucessfully stored basin attributes in" in line:
            full_path = line.split('attributes in ')[1].strip()  # Extract the full path
            run_dir = os.path.dirname(full_path)

    run_command = f"python3 main.py --gpu={gpu} --run_dir={run_dir} --epoch_num={epoch_num} evaluate"
    os.system(run_command)

    # grab the test output file for this split
    file_seed = run_dir.split('seed')[1]
    results_file = glob.glob(f"{run_dir}/*ssm*seed{file_seed}_epoch{epoch_num}.p")[0]
    with open(results_file, 'rb') as f:
        seed_dict = pickle.load(f)

    # create the ensemble dictionary
    for basin in seed_dict:
        seed_dict[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
    return seed_dict, seed

def merge_dicts(dict1, dict2, seed):
    for basin in dict2:
        if basin in dict1:
            dict1[basin] = pd.merge(
                dict1[basin],
                dict2[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)
        else:
            dict1[basin] = dict2[basin]
    return dict1

if __name__ == "__main__":
    with Pool(processes=8) as pool:
        seeds_gpus = [(seed, seed % 8) for seed in range(firstSeed, firstSeed + nSeeds)]
        results = pool.map(run_evaluation, seeds_gpus)

    # Combine the results into a single dictionary
    ens_dict, _ = results[0]
    for result in results[1:]:
        seed_dict, seed = result
        ens_dict = merge_dicts(ens_dict, seed_dict, seed)

    # calculate ensemble mean
    for basin in ens_dict:
        simdf = ens_dict[basin].filter(regex='qsim_')
        ensMean = simdf.mean(axis=1)
        ens_dict[basin].insert(0, 'qsim', ensMean)

    # save the ensemble results as a pickle
    fname = f"analysis/results_data/{experiment}_{note}_{epoch_num}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
