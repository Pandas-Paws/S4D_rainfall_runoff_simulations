import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
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
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091999', format='%d%m%Y')
}

###############
# Helper fn.  #
###############
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

###############
# Arg Parsing #
###############
def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness", "create_splits"])
    parser.add_argument('--camels_root', type=str, default='/scratch/yihan/camels_us_data/basin_dataset_public_v1p2/',
                        help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="User-selected GPU ID - if none chosen, will default to cpu")
    parser.add_argument('--cache_data', type=str2bool, default=True,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers', type=int, default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--no_static', type=str2bool, default=False,
                        help="If True, trains LSTM without static features")
    parser.add_argument('--concat_static', type=str2bool, default=False,
                        help="If True, trains LSTM with static feats at each time step")
    parser.add_argument('--model_name', type=str, default='ssm',
                        help="Choose between ['lstm','ssm'].")
    parser.add_argument('--use_mse', type=str2bool, default=False,
                        help="If True, uses MSE as loss function.")
    parser.add_argument('--n_splits', type=int, default=None,
                        help="Number of splits to create for cross validation (create_splits mode)")
    parser.add_argument('--basin_file', type=str, default=None,
                        help="Text file listing USGS basin IDs (one per line)")
    parser.add_argument('--split_file', type=str, default=None,
                        help="Path to pickle file of CV splits")
    parser.add_argument('--split', type=int, default=12,
                        help="Index of fold to use for train/test in CV")
    parser.add_argument('--epoch_num', type=int, default=30,
                        help="Epoch number to evaluate (evaluate mode)")
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
    parser.add_argument('--epochs', default=30, type=int, help='Training epochs')
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

    # Mode-specific validations
    if cfg['mode'] == 'create_splits':
        if cfg['n_splits'] is None or cfg['split_file'] is None:
            parser.error("create_splits mode requires --n_splits and --split_file.")
    if cfg['mode'] in ['train', 'evaluate'] and cfg['split_file'] is not None and cfg['split'] is None:
        parser.error(f"{cfg['mode']} mode with --split_file requires --split.")

    # Seed
    if cfg['mode'] in ['train', 'create_splits'] and cfg['seed'] is None:
        cfg['seed'] = int(np.random.uniform(low=0, high=1e6))

    # Device
    if cfg['gpu'] >= 0:
        device = f"cuda:{cfg['gpu']}"
    else:
        device = 'cpu'
    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Global defaults
    for key, val in GLOBAL_SETTINGS.items():
        cfg.setdefault(key, val)

    # Paths
    if cfg['camels_root']:
        cfg['camels_root'] = Path(cfg['camels_root'])
    if cfg.get('split_file'):
        cfg['split_file'] = Path(cfg['split_file'])
    if cfg.get('run_dir'):
        cfg['run_dir'] = Path(cfg['run_dir'])

    return cfg

######################
# Cross-Validation   #
######################
def create_splits(cfg: Dict):
    """Create and pickle k-fold train/test splits over basins."""
    if cfg['basin_file']:
        basins = [line.strip() for line in open(cfg['basin_file'])]
    else:
        basins = get_basin_list()

    kf = KFold(n_splits=cfg['n_splits'], shuffle=True, random_state=cfg['seed'])
    splits = {}
    for idx, (train_idx, test_idx) in enumerate(kf.split(basins)):
        splits[idx] = {
            'train': [basins[i] for i in train_idx],
            'test': [basins[i] for i in test_idx]
        }

    with open(cfg['split_file'], 'wb') as f:
        pickle.dump(splits, f)
    print(f"Saved {cfg['n_splits']}-fold splits to {cfg['split_file']}")


###########################
# Setup & Data Prep      #
###########################
def _setup_run(cfg: Dict) -> Dict:
    now = datetime.now()
    
    # include split, model name, and seconds for uniqueness
    timestamp = now.strftime("%d%m_%H%M%S")
    split = cfg.get("split", "all")
    run_name = f'{cfg["model_name"]}_split{split}_{timestamp}_seed{cfg["seed"]}'
    
    
    #run_name = f"run_{now:%d%m}_{now:%H%M}_seed{cfg['seed']}"
    cfg['run_dir'] = Path(__file__).absolute().parent / 'runs' / run_name
    cfg['train_dir'] = cfg['run_dir'] / 'data' / 'train'
    cfg['val_dir']   = cfg['run_dir'] / 'data' / 'val'
    cfg['train_dir'].mkdir(parents=True, exist_ok=True)
    cfg['val_dir'].mkdir(parents=True, exist_ok=True)
    with (cfg['run_dir'] / 'cfg.json').open('w') as fp:
        temp = {k: (str(v) if isinstance(v, (Path, PosixPath)) else 
                  (v.strftime('%d%m%Y') if isinstance(v, pd.Timestamp) else v))
                for k, v in cfg.items()}
        json.dump(temp, fp, indent=4, sort_keys=True)
    return cfg


def _prepare_data(cfg: Dict, basins: List[str]) -> Dict:
    cfg['db_path'] = str(cfg['run_dir'] / 'attributes.db')
    add_camels_attributes(cfg['camels_root'], db_path=cfg['db_path'])
    cfg['train_file'] = cfg['train_dir'] / 'train_data.h5'
    create_h5_files(
        camels_root=cfg['camels_root'],
        out_file=cfg['train_file'],
        basins=basins,
        dates=[cfg['train_start'], cfg['train_end']],
        with_basin_str=True,
        seq_length=cfg['seq_length']
    )
    return cfg

###########################
# Training & Evaluation  #
###########################
def train(cfg: Dict):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if cfg.get('split_file'):
        with open(cfg['split_file'], 'rb') as fp:
            splits = pickle.load(fp)
        basins = splits[cfg['split']]['train']
    else:
        basins = get_basin_list()

    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg, basins)

    ds = CamelsH5(
        h5_file=cfg['train_file'], basins=basins,
        db_path=cfg['db_path'], concat_static=cfg['concat_static'],
        cache=cfg['cache_data'], no_static=cfg['no_static']
    )
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'])

    # Model & optimizer
    input_size_dyn = 5 if (cfg['no_static'] or not cfg['concat_static']) else 32
    if cfg['model_name']=='lstm':
        model = LSTM_Model(input_size_dyn, cfg['hidden_size'],
                           cfg['initial_forget_gate_bias'], cfg['dropout'],
                           cfg['concat_static'], cfg['no_static']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        scheduler = None
    else:  # SSM
        model = HOPE(d_input=input_size_dyn, d_output=1,
                     d_model=cfg['d_model'], n_layers=cfg['n_layers'],
                     dropout=cfg['ssm_dropout'], cfg=cfg,
                     prenorm=cfg['prenorm']).to(DEVICE)
        optimizer, scheduler = setup_optimizer(
            model, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
            epochs=cfg['epochs_scheduler'], warmup_epochs=cfg['warmup']
        )

    loss_func = nn.MSELoss() if cfg['use_mse'] else NSELoss()
    lr_schedule = {11: 5e-4, 26: 1e-4}

    for epoch in trange(1, cfg['epochs']+1, desc='Epochs'):
        if cfg['model_name']!='ssm' and epoch in lr_schedule:
            for g in optimizer.param_groups:
                g['lr'] = lr_schedule[epoch]
        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, scheduler)
        torch.save(model.state_dict(), str(cfg['run_dir']/f'model_epoch{epoch}.pt'))


def train_epoch(model, optimizer, loss_func, loader, cfg, epoch, scheduler):
    model.train()
    total_loss = 0.0
    for x, y, qstd in loader:
        optimizer.zero_grad()
        x, y, qstd = x.to(DEVICE), y.to(DEVICE), qstd.to(DEVICE)
        preds = model(x) if cfg['model_name']=='ssm' else model(x)[0]
        loss = loss_func(preds, y) if cfg['use_mse'] else loss_func(preds, y, qstd)
        loss.backward()
        if cfg['clip_norm']:
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_value'])
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss/len(loader)
    print(f"Epoch {epoch} avg loss {avg:.5f}")
    if scheduler: scheduler.step()


def evaluate(cfg: Dict):
    with open(cfg['run_dir']/ 'cfg.json') as fp:
        run_cfg = json.load(fp)

    if run_cfg.get('split_file'):
        with open(run_cfg['split_file'], 'rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg['split']]['test']
    else:
        basins = get_basin_list()

    train_file = cfg['run_dir']/ 'data'/ 'train'/ 'train_data.h5'
    db_path = str(cfg['run_dir']/ 'attributes.db')
    ds_train = CamelsH5(h5_file=train_file, basins=basins,
                        db_path=db_path, concat_static=run_cfg['concat_static'])
    means, stds = ds_train.get_attribute_means(), ds_train.get_attribute_stds()

    input_size_dyn = 5 if (run_cfg['no_static'] or not run_cfg['concat_static']) else 32
    if run_cfg['model_name']=='lstm':
        model = LSTM_Model(input_size_dyn, run_cfg['hidden_size'],
                           run_cfg['dropout'], run_cfg['concat_static'],
                           run_cfg['no_static']).to(DEVICE)
    else:
        model = HOPE(d_input=input_size_dyn, d_output=1, d_model=run_cfg['d_model'],
                     n_layers=run_cfg['n_layers'], dropout=run_cfg['ssm_dropout'],
                     cfg=run_cfg, prenorm=run_cfg['prenorm']).to(DEVICE)
    model.load_state_dict(torch.load(str(cfg['run_dir']/f"model_epoch{cfg['epoch_num']}.pt"), map_location=DEVICE))

    loss_func = nn.MSELoss() if cfg['use_mse'] else NSELoss()
    results = {}
    total_eval = 0.0
    for basin in tqdm(basins):
        ds_test = CamelsTXT(camels_root=cfg['camels_root'], basin=basin,
                            dates=[GLOBAL_SETTINGS['val_start'], GLOBAL_SETTINGS['val_end']],
                            is_train=False, seq_length=run_cfg['seq_length'],
                            with_attributes=True, attribute_means=means,
                            attribute_stds=stds, concat_static=run_cfg['concat_static'],
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=12)
        preds, obs, eval_loss = evaluate_basin(run_cfg['model_name'], model, loader, loss_func)
        df = pd.DataFrame({'qobs': obs.flatten(), 'qsim': preds.flatten()},
                          index=pd.date_range(start=GLOBAL_SETTINGS['val_start'], end=GLOBAL_SETTINGS['val_end']))
        results[basin] = df
        total_eval += eval_loss
    print(f"Average test loss: {total_eval/len(basins):.5f}")
    _store_results(cfg, run_cfg, results)


def evaluate_basin(model_name, model, loader, loss_func):
    model.eval()
    preds, obs = None, None
    eval_loss = 0.0
    with torch.no_grad():
        for x, y, qstd in loader:
            x, y, qstd = x.to(DEVICE), y.to(DEVICE), qstd.to(DEVICE)
            p = model(x) if model_name=='ssm' else model(x)[0]
            batch_loss = loss_func(p, y) if isinstance(loss_func, nn.MSELoss) else loss_func(p, y, qstd)
            eval_loss += batch_loss.item()
            p_cpu, y_cpu = p.detach().cpu(), y.detach().cpu()
            preds = p_cpu if preds is None else torch.cat((preds, p_cpu), 0)
            obs   = y_cpu if obs   is None else torch.cat((obs, y_cpu), 0)
    preds = rescale_features(preds.numpy(), 'output') if model_name!='mclstm' else preds.numpy()
    preds[preds < 0] = 0
    return preds, obs.numpy(), eval_loss/len(loader)


def _store_results(user_cfg, run_cfg, results: Dict[str, pd.DataFrame]):
    fname = f"{run_cfg['model_name']}_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"
    out = user_cfg['run_dir']/ fname
    with open(out, 'wb') as fp:
        pickle.dump(results, fp)
    print(f"Stored results at {out}")

###########################
# Main                   #
###########################
if __name__ == '__main__':
    cfg = get_args()
    if cfg['mode'] == 'create_splits':
        create_splits(cfg)
        sys.exit(0)
    else:
        globals()[cfg['mode']](cfg)
