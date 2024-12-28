"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd
import scipy.io as mlio

class HOPE_SSM_Kernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.0001, dt_max=0.1, lr=None, lr_dt=None, wd=None):
        super().__init__()
        # Generate dt
        
        log_dt = torch.rand(d_model) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)


        H = torch.randn(d_model, N // 2, dtype=torch.cfloat) / math.sqrt(N) / 2

        self.n = N // 2
        self.h = d_model
        self.register("log_dt", log_dt, 0, lr_dt)
        self.register("H", torch.view_as_real(H), wd, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        samplers = torch.exp(2j * math.pi * repeat(torch.arange(L, device=dt.device) / L, 'n -> h n', h=self.h))
        dt_expanded = dt.unsqueeze(-1)
        samplers_dt = ((1 + dt_expanded) * samplers + dt_expanded - 1) / ((-1 + dt_expanded) * samplers + dt_expanded + 1)
        samplers_angle = torch.angle(samplers_dt) # Create the angles of nonuniform sampling points

        enumer = -torch.arange(self.n, dtype=samplers_angle.dtype, device=samplers_angle.device) - 1  # Ensure K is the same dtype and device as H
        samplers_angle = samplers_angle.unsqueeze(-1)
        samplers_angle = torch.exp(1j * samplers_angle * enumer) # (H L N) Create the angles of nonuniform sampling points
        K = torch.einsum('hn, hln -> hl', torch.view_as_complex(self.H), samplers_angle) # Sample the transfer function

        return K

    def register(self, name, tensor, wd, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": wd}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class HOPE_SSM(nn.Module):
    def __init__(self, d_model, d_state=256, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))
        
        print(self.h, self.n)

        # SSM Kernel
        self.kernel = HOPE_SSM_Kernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        
        # k_f = torch.fft.fft(k) # (H L)
        u_f = torch.fft.fft(u) # (B H L)
        y = torch.fft.ifft(u_f*k).real # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
