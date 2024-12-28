import math
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import tqdm
from typing import Tuple

import copy


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from models.s4.s4d import S4D as LTI

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d
    
#from tqdm.auto import tqdm
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.cuda.FloatTensor(x.shape).uniform_()-1) + 1 
    return mult_noise * x + add_noise     


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights
    

def warmup_cosine_annealing_lr(epoch, warmup_epochs, total_epochs, lr_start=1.0, lr_end=0.05):
    if epoch < warmup_epochs:
        lr = lr_start * (epoch / warmup_epochs)
    else:
        lr = lr_end + (lr_start - lr_end) * 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )
    return lr


class HOPE(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.1,
        cfg=None,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                LTI(d_model, dropout=dropout, transposed=True,
                         lr=min(cfg["lr_min"], cfg["lr"]), d_state=cfg["d_state"], dt_min=cfg["min_dt"], dt_max=cfg["max_dt"], lr_dt=cfg["lr_dt"], cfr=cfg["cfr"], cfi=cfg["cfi"], wd=cfg["wd"])
            )
            #self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        #self.lnorm = torch.nn.LayerNorm(365)
        #self.att = SelfAttentionLayer(365) #20240626 remove attention


    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # x = torch.cat((x, torch.zeros((x.shape[0],1,x.shape[2])).to('cuda')), 1 )
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        # x = torch.cat((x,torch.flip(x,dims=[-1])),dim=-1) # bi-directional
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x

            #z, _ = self.att(z)
            
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x    
    
def setup_optimizer(model, lr, weight_decay, epochs, warmup_epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Define lambda function for the scheduler
    lr_lambda = lambda epoch: warmup_cosine_annealing_lr(epoch, warmup_epochs, epochs, lr_start=1.0, lr_end=0.0)

    # Create the scheduler with LambdaLR
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler    


#export CUDA_VISIBLE_DEVICES=1; python3 SSM_test.py --epochs=200 --d_model=128  --lr=0.0001 --lr_min=0.00001 --weight_decay=0.01 --wd=0.01  --lr_dt=0.001 --min_dt=0.001 --max_dt=0.1  --epochs_scheduler=200 --warmup=10 --seed 1 --n_layer 6
