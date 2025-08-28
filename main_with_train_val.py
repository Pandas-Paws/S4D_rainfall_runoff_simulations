import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
import pdb
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import add_camels_attributes, rescale_features
from papercode.lstm import LSTM_Model
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list
from papercode.SSM_test import HOPE, setup_optimizer


###########
# Globals #
###########

GLOBAL_SETTINGS = {
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'hidden_size': 256,
    'initial_forget_gate_bias': 3,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 365,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101984', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091989', format='%d%m%Y'),
    'test_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'test_end': pd.to_datetime('30091999', format='%d%m%Y')
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness", "create_splits"])
    parser.add_argument(
        '--camels_root',
        type=str,
        default='/scratch/yihan/camels_us_data/basin_dataset_public_v1p2/',
        help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help="User-selected GPU ID - if none chosen, will default to cpu")
    parser.add_argument(
        '--cache_data', type=str2bool, default=True, help="If True, loads all data into memory")
    parser.add_argument(
        '--num_workers', type=int, default=12, help="Number of parallel threads for data loading")
    parser.add_argument(
        '--no_static',
        type=str2bool,
        default=False,
        help="If True, trains LSTM without static features")
    parser.add_argument(
        '--concat_static',
        type=str2bool,
        default=False,
        help="If True, train LSTM with static feats concatenated at each time step")
    # -- start --
    # yhwang 20240604
    parser.add_argument(
        '--model_name',
        type=str,
        default='ssm',
        help="Choose between ['lstm','ssm'].")
    # -- end --
    parser.add_argument(
        '--use_mse',
        type=str2bool,
        default=False,
        help="If True, uses mean squared error as loss function.")
    parser.add_argument(
        '--basin_file',
        type=str,
        default=None,
        help="Path to file containing usgs basin ids. Default is data/basin_list.txt")
    parser.add_argument(
        '--split_file',
        type=str,
        default=None,
        help="Path to file created from the `create_splits` function.")
    parser.add_argument('--epoch_num', type=int, default=30, help="Epoch number to evaluate")
        
    
    #====================================#
    # Argument for SSM    
    # Optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--lr_min', default=0.001, type=float, help='SSM Learning rate')
    parser.add_argument('--lr_dt', default=0.0, type=float, help='dt lr')
    parser.add_argument('--min_dt', default=0.001, type=float, help='min dt')
    parser.add_argument('--max_dt', default=1, type=float, help='max dt')
    parser.add_argument('--wd', default=0.0, type=float, help='H weight decay')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    
    # Scheduler
    parser.add_argument('--epochs', default=50, type=int, help='Training epochs')
    parser.add_argument('--epochs_scheduler', default=50, type=int, help='Total epochs for scheduler')
    parser.add_argument('--warmup', default=10, type=int, help='warmup epochs')
    
    # Dataloader
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=64, type=int, help='Model dimension')
    parser.add_argument('--ssm_dropout', default=0.15, type=float, help='Dropout')
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')

    parser.add_argument('--d_state', default=64, type=int)
    parser.add_argument('--cfr', default=1.0, type=float)
    parser.add_argument('--cfi', default=1.0, type=float)
    
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--model', type=str, default='s4d', metavar='N', help='model name')
    #====================================#

    cfg = vars(parser.parse_args())
    
    # Validation checks
    if (cfg["mode"] in ["train", "create_splits"]) and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # GPU selection
    if cfg["gpu"] >= 0:
        device = f"cuda:{cfg['gpu']}"
    else:
        device = 'cpu'

    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    # combine global settings with user config
    #cfg.update(GLOBAL_SETTINGS)
    for key, value in GLOBAL_SETTINGS.items():
        cfg.setdefault(key, value)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    now = datetime.now()
    run_name = f"run_{now:%d%m}_{now:%H%M}_seed{cfg['seed']}"
    cfg['run_dir'] = Path(__file__).parent / "runs" / run_name
    if cfg['run_dir'].exists():
        raise RuntimeError(f"Run dir already exists: {cfg['run_dir']}")
    (cfg['run_dir'] / 'data' / 'train').mkdir(parents=True)
    (cfg['run_dir'] / 'data' / 'val').mkdir(parents=True)
    with open(cfg['run_dir'] / 'cfg.json', 'w') as fp:
        serial = {}
        for k, v in cfg.items():
            if isinstance(v, (PosixPath, Path)):
                serial[k] = str(v)
            elif isinstance(v, pd.Timestamp):
                serial[k] = v.strftime("%d%m%Y")
            else:
                serial[k] = v
        json.dump(serial, fp, indent=4, sort_keys=True)
    return cfg


def _prepare_data(cfg: Dict, basins: List[str]) -> Dict:
    cfg['db_path'] = str(cfg['run_dir'] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg['db_path'])

    cfg['train_file'] = cfg['run_dir'] / 'data' / 'train' / 'train_data.h5'
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=cfg['train_file'],
        basins=basins,
        dates=[cfg['train_start'], cfg['train_end']],
        with_basin_str=True,
        seq_length=cfg['seq_length']
    )
    
    '''
    cfg['val_file'] = cfg['run_dir'] / 'data' / 'val' / 'val_data.h5'
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=cfg['val_file'],
        basins=basins,
        dates=[cfg['val_start'], cfg['val_end']],
        with_basin_str=True,
        seq_length=cfg['seq_length']
    )
    '''
    return cfg


def train(cfg: Dict):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    if cfg["split_file"]:
        with open(cfg["split_file"], 'rb') as fp:
            splits = pickle.load(fp)
        basins = splits[cfg["split"]]["train"]
    else:
        basins = get_basin_list()

    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg, basins)

    # full training dataset
    full_ds = CamelsH5(
        h5_file=cfg["train_file"],
        basins=basins,
        db_path=cfg["db_path"],
        concat_static=cfg["concat_static"],
        cache=cfg["cache_data"],
        no_static=cfg["no_static"]
    )

    # split into 90% train / 10% val
    n_total = len(full_ds)
    n_val = int(n_total * 0.1) # todo, 0.1
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    
    '''
    ######### this use 5-year validation set instead of ##########
    ######### using 10% of 10-year train data ####################
    train_ds =full_ds
    val_ds   = CamelsH5(
        h5_file=str(cfg['val_file']),
        basins=basins,
        db_path=cfg['db_path'],
        concat_static=cfg["concat_static"],
        cache=cfg["cache_data"],
        no_static=cfg["no_static"]
    )
    '''
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=cfg["num_workers"])

    # build model & optimizer
    inp_dyn = 5 if (cfg["no_static"] or not cfg["concat_static"]) else 32
    if cfg["model_name"] == 'lstm':
        model = LSTM_Model(
            input_size_dyn=inp_dyn,
            hidden_size=cfg["hidden_size"],
            initial_forget_bias=cfg["initial_forget_gate_bias"],
            dropout=cfg["dropout"],
            concat_static=cfg["concat_static"],
            no_static=cfg["no_static"]
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        scheduler = None
    else:
        model = HOPE(
            d_input=inp_dyn,
            d_output=1,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            dropout=cfg["ssm_dropout"],
            cfg=cfg,
            prenorm=cfg["prenorm"]
        ).to(DEVICE)
        optimizer, scheduler = setup_optimizer(
            model,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            epochs=cfg["epochs_scheduler"],
            warmup_epochs=cfg["warmup"]
        )

    loss_fn = nn.MSELoss() if cfg["use_mse"] else NSELoss()
    lr_schedule = {11: 5e-4, 26: 1e-4}

    for epoch in trange(1, cfg["epochs"] + 1, desc="Epochs"):
        # adjust non-SSM LR
        if cfg["model_name"] != 'ssm' and epoch in lr_schedule:
            for pg in optimizer.param_groups:
                pg['lr'] = lr_schedule[epoch]

        # train pass
        model.train()
        total_loss = 0.0
        for x, y, q_std in train_loader:
            x, y, q_std = x.to(DEVICE), y.to(DEVICE), q_std.to(DEVICE)
            optimizer.zero_grad()
            out = model(x) if cfg["model_name"] == 'ssm' else model(x)[0]
            loss = loss_fn(out, y) if cfg["use_mse"] else loss_fn(out, y, q_std)
            loss.backward()
            if cfg["clip_norm"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # val pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, q_std in val_loader:
                x, y, q_std = x.to(DEVICE), y.to(DEVICE), q_std.to(DEVICE)
                out = model(x) if cfg["model_name"] == 'ssm' else model(x)[0]
                l = loss_fn(out, y) if cfg["use_mse"] else loss_fn(out, y, q_std)
                val_loss += l.item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}  Train loss: {avg_train:.5f}  Val loss: {avg_val:.5f}")

        # save checkpoint
        torch.save(model.state_dict(), cfg["run_dir"] / f"model_epoch{epoch}.pt")

        if scheduler is not None:
            scheduler.step()


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if user_cfg["split_file"] is not None:
        with Path(user_cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg["split"]]["test"]
    else:
        basins = get_basin_list()

    # get attribute means/stds from trainings dataset
    train_file = user_cfg["run_dir"] / "data/train/train_data.h5"
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    ds_train = CamelsH5(
        h5_file=train_file, db_path=db_path, basins=basins, concat_static=run_cfg["concat_static"])
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()

    # create model
    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 32
    if run_cfg["model_name"] == 'lstm':
        model = LSTM_Model(
            input_size_dyn=input_size_dyn,
            hidden_size=run_cfg["hidden_size"],
            dropout=run_cfg["dropout"],
            concat_static=run_cfg["concat_static"],
            no_static=run_cfg["no_static"]).to(DEVICE)
            
    elif run_cfg["model_name"] == 'ssm':
        model = HOPE(
            d_input=input_size_dyn,
            d_output=1,
            d_model=run_cfg["d_model"],
            n_layers=run_cfg["n_layers"],
            dropout=run_cfg["ssm_dropout"],
            cfg = run_cfg, 
            prenorm=run_cfg["prenorm"],
        ).to(DEVICE)

    # load trained model
    epoch_num = user_cfg["epoch_num"]
    weight_file = user_cfg["run_dir"] / f'model_epoch{epoch_num}.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
    
    if user_cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()
    

    date_range = pd.date_range(start=GLOBAL_SETTINGS["test_start"], end=GLOBAL_SETTINGS["test_end"])
    results = {}
    
    eval_loss_ssm = 0
    for basin in tqdm(basins):
        ds_test = CamelsTXT(
            camels_root=user_cfg["camels_root"],
            basin=basin,
            dates=[GLOBAL_SETTINGS["test_start"], GLOBAL_SETTINGS["test_end"]],
            is_train=False,
            seq_length=run_cfg["seq_length"],
            with_attributes=True,
            attribute_means=means,
            attribute_stds=stds,
            concat_static=run_cfg["concat_static"],
            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
        
        if run_cfg["model_name"] == 'lstm':
            preds, obs = evaluate_basin(run_cfg['model_name'], model, loader, loss_func)
            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
        elif run_cfg["model_name"] == 'ssm':
            preds, obs, avg_eval_loss = evaluate_basin(run_cfg['model_name'], model, loader, loss_func)
            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
            eval_loss_ssm += avg_eval_loss
        
        results[basin] = df
        
    # Save the evaluation loss to the text file
    loss_file = "train_and_test_loss.txt"
    loss = eval_loss_ssm / 531
    with open(loss_file, "a") as f:
        f.write(f"Average Evaluation Loss Epoch {epoch_num}: {loss:.5f}\n")

    _store_results(user_cfg, run_cfg, results)


def evaluate_basin(model_name: str, model: nn.Module, loader: DataLoader, loss_func: nn.Module, use_mse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None
    trash_cell, cell_state = None, None
    
    eval_loss = 0.0  # Initialize evaluation loss
    num_batches = len(loader) 
    
    with torch.no_grad():
        for data in loader:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            if model_name == "lstm":
                p = model(x)[0]
            elif model_name == "ssm":
                p = model(x)
        
            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)
                
            # Compute loss for the current batch
            if use_mse:
                loss = loss_func(p, y)
            else:
                q_stds = q_stds.to(DEVICE)
                loss = loss_func(p, y, q_stds)
            
            eval_loss += loss.item()
        if model_name != 'mclstm':
            preds = rescale_features(preds.numpy(), variable='output')
        else:
            preds = preds.numpy()
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0
        
        avg_eval_loss = eval_loss / num_batches
        
        if trash_cell is not None:
            trash_cell = trash_cell.numpy()
            cell_state = cell_state.numpy()
            return preds, obs, trash_cell, cell_state
            
    return preds, obs, avg_eval_loss


def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"{run_cfg['model_name']}_no_static_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"{run_cfg['model_name']}_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"
        else:
            file_name = user_cfg["run_dir"] / f"ealstm_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
