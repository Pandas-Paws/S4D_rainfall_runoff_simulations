#!/usr/bin/env python3
"""
Run evaluation across seeds and k-folds in parallel (capped at 3 jobs per GPU)
and merge the per-basin predictions into one PUB ensemble.

Usage
-----
    python run_pub_parallel.py <experiment> <note> <epoch_num>

Example
-------
    python run_pub_parallel.py ssm test 30
"""
import glob
import os
import pickle
import sys
from multiprocessing import Pool
import pdb
import pandas as pd

# ---------------------------------------------------------------------#
# Configuration                                                        #
# ---------------------------------------------------------------------#
nsplits       = 12
seeds = [200, 201, 202, 203, 204, 205, 206, 207]
nGPUs         = 8
jobs_per_gpu  = 3               # =3 concurrent evaluations per GPU
max_processes = nGPUs * jobs_per_gpu

# ---------------------------------------------------------------------#
# Command-line arguments                                               #
# ---------------------------------------------------------------------#
if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

experiment = sys.argv[1]   # e.g. "ssm", "lstm"
note       = sys.argv[2]   # e.g. "test"
epoch_num  = sys.argv[3]   # e.g. "50" ("30" for lstm)

# ---------------------------------------------------------------------#
# Helper functions                                                     #
# ---------------------------------------------------------------------#
def run_evaluation(args):
    """
    Run evaluation for one (seed, split) on the assigned GPU,
    skipping if the desired pickle already exists.
    """
    seed, split, gpu = args
    report_file = f"reports/pub_{experiment}.{seed}.{split}.out"
    print(f"[seed={seed} split={split}] parsing {report_file}")

    # 1) Extract run_dir from the report
    run_dir = None
    with open(report_file, "r") as f:
        for line in f:
            if "Sucessfully stored basin attributes in" in line:
                full_path = line.strip().split(" in ", 1)[1].rstrip(".")
                run_dir = os.path.dirname(full_path)
                break
    if run_dir is None:
        raise RuntimeError(f"Run directory not found in report: {report_file}")

    # 2) Check for existing pickle
    pattern = f"{run_dir}/*{experiment}*seed{seed}_epoch{epoch_num}.p"
    matches = glob.glob(pattern)
    if matches:
        results_file = matches[0]
        print(f"[seed={seed} split={split}] cached ? {results_file}")
    else:
        print(f"[seed={seed} split={split}] no cache ? evaluating on GPU {gpu}")
        split_file = f"data/kfold_splits_seed{seed}.p"
        cmd = (
            f"python3 main_pub.py evaluate "
            f"--gpu={gpu} "
            f"--run_dir={run_dir} "
            f"--epoch_num={epoch_num} "
            f"--split={split} "
            f"--split_file={split_file}"
        )
        os.system(cmd)
        matches = glob.glob(pattern)
        if not matches:
            raise RuntimeError(f"No pickle found after evaluation for seed={seed}, split={split}")
        results_file = matches[0]

    # 3) Load the pickle
    with open(results_file, "rb") as fp:
        seed_dict = pickle.load(fp)

    # 4) Rename qsim column for merging
    colname = f"qsim_s{seed}_f{split}"
    for basin in seed_dict:
        seed_dict[basin].rename(columns={"qsim": colname}, inplace=True)

    return seed_dict, (seed, split)


def merge_dicts(ens, new, key):
    """Merge another seed/split's predictions into the ensemble dict."""
    col = f"qsim_s{key[0]}_f{key[1]}"
    for basin, df in new.items():
        if basin in ens:
            ens[basin] = pd.merge(
                ens[basin],
                df[[col]],
                left_index=True,
                right_index=True,
                how="inner",
            )
        else:
            ens[basin] = df
    return ens


# ---------------------------------------------------------------------#
# Main                                                                 #
# ---------------------------------------------------------------------#
if __name__ == "__main__":
    # Build list of (seed, split, gpu) tasks
    tasks = []
    for seed in seeds:
        for split in range(nsplits):
            # round-robin GPU assignment
            gpu = (seed * nsplits + split) % nGPUs
            tasks.append((seed, split, gpu))

    print(f"Launching up to {max_processes} parallel evaluations "
          f"({jobs_per_gpu} per GPU) out of {len(tasks)} total tasks.")

    # Run evaluations in parallel, capped at max_processes
    with Pool(processes=max_processes) as pool:
        results = pool.map(run_evaluation, tasks)

    # Merge all results into one ensemble dict
    ens_dict, _ = results[0]
    for seed_dict, key in results[1:]:
        ens_dict = merge_dicts(ens_dict, seed_dict, key)

    # Compute ensemble mean across qsim columns
    for basin, df in ens_dict.items():
        sim_cols = df.filter(regex=r"^qsim_s\d+_f\d+$")
        df.insert(0, "qsim", sim_cols.mean(axis=1))

    # Save the final ensemble
    os.makedirs("analysis/results_data", exist_ok=True)
    out_path = f"analysis/results_data/{experiment}_{note}_{epoch_num}.pkl"
    with open(out_path, "wb") as fp:
        pickle.dump(ens_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nEnsemble stored at {out_path}")
