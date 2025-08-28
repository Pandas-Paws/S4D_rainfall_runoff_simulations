#!/usr/bin/env python3
"""
Run (or reuse) evaluation for multiple seeds in parallel and merge the
per-basin predictions into an ensemble.

Usage
-----
python run_global_paralle.py <experiment> <note> <epoch_num>

"""
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import pandas as pd

# ---------------------------------------------------------------------#
# Configuration                                                        #
# ---------------------------------------------------------------------#
nSeeds     = 8
firstSeed  = 200                     # first seed (inclusive)
seeds      = [200, 201, 202, 203, 204, 205, 206, 207]  # explicit list
nGPUs      = 8                       # how many GPUs are available

# ---------------------------------------------------------------------#
# Command-line arguments                                               #
# ---------------------------------------------------------------------#
experiment = sys.argv[1]             # e.g. "ssm"
note       = sys.argv[2]             # e.g. "test"
epoch_num  = sys.argv[3]             # e.g. "30"


# ---------------------------------------------------------------------#
# Helper functions                                                     #
# ---------------------------------------------------------------------#
def run_evaluation(seed_gpu):
    """Run evaluation for one seed on the assigned GPU — but skip if the
    desired pickle already exists.

    Returns
    -------
    (seed_dict, seed)
        * seed_dict: {basin_id: DataFrame(qobs, qsim_XXX)}
        * seed:      the seed integer (used later when merging)
    """
    seed, gpu = seed_gpu
    report_file = f"reports/{experiment}.{seed}_{note}.out"
    print(f"[seed {seed}] → parsing report {report_file}")

    # ------------------------------------------------------------------
    # 1) Figure out this seed's run directory from the screen report
    # ------------------------------------------------------------------
    with open(report_file, "r") as f:
        for line in f:
            if "Sucessfully stored basin attributes in" in line:
                run_dir = os.path.dirname(line.split("attributes in ")[1].strip())
                break
        else:
            raise RuntimeError("Run directory not found in report!")

    # ------------------------------------------------------------------
    # 2) Check if the results pickle for this epoch already exists
    # ------------------------------------------------------------------
    results_glob = glob.glob(f"{run_dir}/*ssm*seed{seed}_epoch{epoch_num}.p")
    if results_glob:
        results_file = results_glob[0]
        print(f"[seed {seed}] -> cached pickle found → {results_file}")
    else:
        print(f"[seed {seed}] -> no cache → running evaluation on GPU {gpu}")
        cmd = (
            f"python3 main.py --gpu={gpu} --run_dir={run_dir} "
            f"--epoch_num={epoch_num} evaluate"
        )
        os.system(cmd)
        results_glob = glob.glob(f"{run_dir}/*ssm*seed{seed}_epoch{epoch_num}.p")
        if not results_glob:
            raise RuntimeError(f"Expected pickle not found after evaluation for seed {seed}")
        results_file = results_glob[0]

    # ------------------------------------------------------------------
    # 3) Load the pickle
    # ------------------------------------------------------------------
    with open(results_file, "rb") as f:
        seed_dict = pickle.load(f)

    # Rename qsim column so we can concatenate across seeds later
    for basin in seed_dict:
        seed_dict[basin].rename(columns={"qsim": f"qsim_{seed}"}, inplace=True)

    return seed_dict, seed


def merge_dicts(dict1, dict2, seed):
    """Merge another seed's predictions into the ensemble dict."""
    for basin in dict2:
        if basin in dict1:
            dict1[basin] = pd.merge(
                dict1[basin],
                dict2[basin][f"qsim_{seed}"],       # only the new column
                left_index=True,
                right_index=True,
                how="inner",
            )
        else:
            dict1[basin] = dict2[basin]
    return dict1


# ---------------------------------------------------------------------#
# Main                                                                 #
# ---------------------------------------------------------------------#
if __name__ == "__main__":

    # Launch seeds in parallel (one GPU per seed modulo nGPUs)
    seeds_gpus = [(seed, seed % nGPUs) for seed in range(firstSeed, firstSeed + nSeeds)]

    with Pool(processes=nSeeds) as pool:
        results = pool.map(run_evaluation, seeds_gpus)

    # ------------------------------------------------------------------
    # Combine per-seed DataFrames into an ensemble dictionary
    # ------------------------------------------------------------------
    ens_dict, _ = results[0]
    for seed_dict, seed in results[1:]:
        ens_dict = merge_dicts(ens_dict, seed_dict, seed)

    # ------------------------------------------------------------------
    # Compute ensemble mean for each basin
    # ------------------------------------------------------------------
    for basin in ens_dict:
        simdf = ens_dict[basin].filter(regex=r"^qsim_\d+$")
        ens_dict[basin].insert(0, "qsim", simdf.mean(axis=1))

    # ------------------------------------------------------------------
    # Save the ensemble results
    # ------------------------------------------------------------------
    out_path = f"analysis/results_data/{experiment}_{note}_{epoch_num}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nEnsemble stored at {out_path}")
