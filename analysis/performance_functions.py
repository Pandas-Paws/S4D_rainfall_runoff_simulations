import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from typing import Tuple

def nse(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> float:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    SSerr = np.mean(np.square(df[sim_col][idex] - df[obs_col][idex]))
    SStot = np.mean(np.square(df[obs_col][idex] - np.mean(df[obs_col][idex])))
    return 1 - SSerr / SStot

def alpha_nse(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> float:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df[sim_col][idex]
    qobs = df[obs_col][idex]
    std_qsim = np.std(qsim)
    std_qobs = np.std(qobs)
    alpha = std_qsim / std_qobs
    return alpha

def beta_nse(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> float:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df[sim_col][idex]
    qobs = df[obs_col][idex]
    mean_qsim = np.mean(qsim)
    mean_qobs = np.mean(qobs)
    beta = (mean_qsim - mean_qobs) / np.std(qobs)
    return beta

def kge(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[float, float, float, float]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df[sim_col][idex]
    qobs = df[obs_col][idex]
    
    r = np.corrcoef(qsim, qobs)[0, 1]
    alpha = np.std(qsim) / np.std(qobs)
    beta = np.mean(qsim) / np.mean(qobs)
    
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge, r, alpha, beta

def get_quant(df: pd.DataFrame, quant: float, seed: int = None, obs_col: str = "qobs") -> Tuple[float, float]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    sim = df[sim_col][idex].quantile(quant)
    obs = df[obs_col][idex].quantile(quant)
    return obs, sim

def bias(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> float:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    sim = df[sim_col][idex].mean()
    obs = df[obs_col][idex].mean()
    return (obs - sim) / obs * 100

def stdev_rat(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> float:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    sim = df[sim_col][idex].std()
    obs = df[obs_col][idex].std()
    return sim / obs

def zero_freq(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[int, int]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    sim = (df[sim_col][idex] == 0).astype(int).sum()
    obs = (df[obs_col][idex] == 0).astype(int).sum()
    return obs, sim

def flow_duration_curve(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[float, float]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    obs33, sim33 = get_quant(df, 0.33, seed, obs_col)
    obs66, sim66 = get_quant(df, 0.66, seed, obs_col)
    sim = (np.log(sim33) - np.log(sim66)) / (0.66 - 0.33)
    obs = (np.log(obs33) - np.log(obs66)) / (0.66 - 0.33)
    return obs, sim

def baseflow_index(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[float, float]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    obsQ = df[obs_col][idex].values
    simQ = df[sim_col][idex].values
    nTimes = len(obsQ)

    obsQD = np.full(nTimes, np.nan)
    simQD = np.full(nTimes, np.nan)
    obsQD[0] = obsQ[0]
    simQD[0] = simQ[0]

    c = 0.925
    for t in range(1, nTimes):
        obsQD[t] = c * obsQD[t - 1] + (1 + c) / 2 * (obsQ[t] - obsQ[t - 1])
        simQD[t] = c * simQD[t - 1] + (1 + c) / 2 * (simQ[t] - simQ[t - 1])

    obsQB = obsQ - obsQD
    simQB = simQ - simQD

    obs = np.mean(np.divide(obsQB[1:], obsQ[1:]))
    sim = np.mean(np.divide(simQB[1:], simQ[1:]))
    return obs, sim

def high_flows(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[int, int]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    obsMedian = df[obs_col][idex].median()
    obsFreq = len(df[obs_col][idex].index[(df[obs_col][idex] >= 9 * obsMedian)].tolist())
    simMedian = df[sim_col][idex].median()
    simFreq = len(df[sim_col][idex].index[(df[sim_col][idex] >= 9 * simMedian)].tolist())
    return obsFreq, simFreq

def low_flows(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[int, int]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idex = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    obsMedian = df[obs_col][idex].median()
    obsFreq = len(df[obs_col][idex].index[(df[obs_col][idex] <= 0.2 * obsMedian)].tolist())
    simMedian = df[sim_col][idex].median()
    simFreq = len(df[sim_col][idex].index[(df[sim_col][idex] <= 0.2 * simMedian)].tolist())
    return obsFreq, simFreq

def FHV(df: pd.DataFrame, percentile_value: int, seed: int = None, obs_col: str = "qobs") -> float:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idx = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df.loc[idx, sim_col]
    qobs = df.loc[idx, obs_col]
    
    # Calculate the percentile exceedance threshold
    qsim_threshold = np.percentile(qsim, 100 - percentile_value)
    qobs_threshold = np.percentile(qobs, 100 - percentile_value)
    
    # Identify high flow values
    qsim_high_flows = qsim[qsim > qsim_threshold]
    qobs_high_flows = qobs[qobs > qobs_threshold]
    
    # Calculate mean of high flow values
    qsim_high_mean = qsim_high_flows.mean()
    qobs_high_mean = qobs_high_flows.mean()
    
    # Calculate FHV
    fhv = ((qsim_high_mean - qobs_high_mean) / qobs_high_mean) 
    
    return fhv * 100
    

def _get_fdc(da: DataArray) -> np.ndarray:
    return da.sortby(da, ascending=False).values

def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError("Metrics only defined for time series (1d or 2d with second dimension 1)")

def _mask_valid(obs: DataArray, sim: DataArray) -> Tuple[DataArray, DataArray]:
    # mask of invalid entries. NaNs in simulations can happen during validation/testing
    idx = (~sim.isnull()) & (~obs.isnull())

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim

def FLV(df: pd.DataFrame, seed: int = None, l: float = 0.3, obs_col: str = "qobs") -> float:
    
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idx = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df.loc[idx, sim_col]
    qobs = df.loc[idx, obs_col]
    
    obs = xr.DataArray(qobs, dims=['time'])
    sim = xr.DataArray(qsim, dims=['time'])

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 1:
        return np.nan

    if (l <= 0) or (l >= 1):
        raise ValueError("l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    obs = obs[-np.round(l * len(obs)).astype(int):]
    sim = sim[-np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs)
    sim = np.log(sim)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100

def mass_balance(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> Tuple[float, float, float]:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idx = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df.loc[idx, sim_col]
    qobs = df.loc[idx, obs_col]
    massbias_total = abs(qsim.sum() - qobs.sum()) / qobs.sum()
    if qsim.sum() - qobs.sum() > 0:
        massbias_pos = (qsim.sum() - qobs.sum()) / qobs.sum()
        massbias_neg = 0
    else:
        massbias_neg = - (qsim.sum() - qobs.sum()) / qobs.sum()
        massbias_pos = 0
    return massbias_total * 100, massbias_pos * 100, massbias_neg * 100

def tc_analysis(df: pd.DataFrame, seed: int = None, obs_col: str = "qobs") -> None:
    sim_col = f"qsim_{seed}" if seed is not None else "qsim"
    idx = df.index[(df[sim_col] >= 0) & (df[obs_col] >= 0)].tolist()
    qsim = df.loc[idx, sim_col]
    qobs = df.loc[idx, obs_col]
    if "trashcell" in df.columns:
        qtc_mean = df["trashcell"].mean()
    return
