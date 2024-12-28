"""Implementation of modular block design used in S4. Compatible with other kernels."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from functools import partial
from einops import rearrange, repeat
import scipy.io as mlio

from src.models.nn import LinearActivation, Activation, DropoutNd
from src.models.sequence.base import SequenceModule

import src.utils.train
log = src.utils.train.get_logger(__name__)

contract = torch.einsum

import time

contract = torch.einsum

from typing import Optional, Mapping, Tuple, Union
from collections import defaultdict
import math

log = src.utils.train.get_logger(__name__)

"""SSM convolution kernels.

SSMKernelDPLR is the S4 kernel, implementing the 'diagonal plus low-rank' algorithm from the original S4 paper. This stores parameters A, B, C, dt, and calling it creates the SSM convolution kernel bar{K}.

SSMKernelDense is a much simpler version included for illustration purposes. It has the same output, but uses the naive SSM algorithm which is much slower. This module is meant for testing and exposition, to understand what the SSM Kernel actually does.

SSMKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.

SSMKernel wraps these with common options and handles the initialization.
"""

from torch import Tensor # For type hints
import torch.optim as optim
import numpy as np
import os

import src.models.hippo.hippo as hippo
from src.models.functional.krylov import krylov, power
import src.utils.train
import scipy.io as mlio

dirname = os.path.dirname(__file__)

log = src.utils.train.get_logger(__name__)

# Try CUDA extension
try:
    from extensions.kernels.cauchy import cauchy_mult as cauchy_cuda
    from extensions.kernels.vandermonde import log_vandermonde_cuda
    has_cuda_extension = True
    log.info("CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) found.")
except:
    log.warning(
        "CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) not found. Install by going to extensions/kernels/ and running `python setup.py install`, for improved speed and memory efficiency. Note that the kernel changed for state-spaces 4.0 and must be recompiled."
    )
    has_cuda_extension = False

try:
    import pykeops
    from src.models.functional.cauchy import cauchy_conj as cauchy_keops
    from src.models.functional.vandermonde import log_vandermonde as log_vandermonde_keops, log_vandermonde_transpose as log_vandermonde_transpose_keops

    has_pykeops = True
    log.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        log.warning(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency."
        )

# Fallback versions
from src.models.functional.cauchy import cauchy_naive
from src.models.functional.vandermonde import log_vandermonde_naive
from src.models.functional.vandermonde import log_vandermonde_transpose_naive

# Alias torch.einsum; can easily swap to opt_einsum if desired
contract = torch.einsum

_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

def inv_transform(param, transform='none'):
    """Initialize a (positive) parameter under a transform."""
    param = torch.clamp(param, min=1e-4)
    if transform == 'none':
        return param
    elif transform == 'exp':
        return torch.log(param) # Some of the HiPPO methods have real part 0
    elif transform == 'relu':
        return param
    elif transform == 'sigmoid':
        return torch.logit(param)
    elif transform == 'softplus':
        return torch.log(torch.exp(param)-1)
    else: raise NotImplementedError

def param_transform(param, transform='none'):
    """Get a (positive) parameter under a transform."""
    if transform == 'none':
        p = param
    elif transform == 'exp':
        p = torch.exp(param)
    elif transform == 'relu':
        # JAX version seems to NaN if you allow 0's, although this code was fine without it
        p = F.relu(param)+1e-4
    elif transform == 'sigmoid':
        p = F.sigmoid(param)
    elif transform == 'softplus':
        p = F.softplus(param)
    else: raise NotImplementedError
    return p


class Kernel(nn.Module):
    """Interface for modules that produce convolution kernels.

    A main distinction between these and normal Modules is that the forward pass
    does not take inputs. It is a mapping from parameters to a tensor that can
    be used in other modules, in particular as a convolution kernel.

    Because of the unusual parameterization, these kernels may often want special
    hyperparameter settings on their parameters. The `register` method provides
    an easy interface for controlling this, and is intended to be used with an
    optimizer hook that can be found in train.py or example.py.

    This class also defines an interface for interacting with kernels *statefully*,
    in particular for state space models (SSMs). This interface handles the setting
    when a model can be converted from a "CNN" into an "RNN".
    _setup_step()
    step()
    default_state()
    forward_state()

    See ConvKernel for the simplest instantiation of this interface.
    """

    def __init__(
        self,
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        **kwargs,
    ):
        """General interface.

        d_model (H): Model dimension, or number of independent convolution kernels created.
        channels (C): Extra dimension in the returned output (see .forward()).
            - One interpretation is that it expands the input dimension giving it C separate "heads" per feature.
              That is convolving by this kernel maps shape (B L D) -> (B L C D)
            - This is also used to implement a particular form of bidirectionality in an efficient way.
            - In general for making a more powerful model, instead of increasing C
              it is recommended to set channels=1 and adjust H to control parameters instead.
        l_max (L): Maximum kernel length (optional). If unspecified, most Kernel instantiations
            will return kernels of arbitrary length as passed into .forward().
        lr: Optional dictionary specifying special hyperparameters for .register().
            Passing in a number (e.g. 0.001) sets attributes of SSM parameters (A, B, dt).
            A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        wd: Same as lr, but for weight decay.
        """
        super().__init__()
        assert d_model > 0
        self.H = self.d_model = d_model
        self.L = self.l_max = l_max
        self.channels = channels
        self.lr = lr
        self.wd = wd
        self.verbose = verbose

        # Add a catch-all **kwargs to make it easier to change kernels
        # without manually moving other options passed in the config.
        # Good to log these just so it's explicit.
        if self.verbose and len(kwargs) > 0:
            log.info(f"{type(self)} extra kwargs: {kwargs}")

        # Logic for registering parameters
        # Case 1: lr: None | float
        #   All params should have this lr (None means inherit from global lr)
        # Case 2: lr: dict
        #   Specified params should have that lr, all others should be None
        if self.lr is None or isinstance(self.lr, float):
            self.lr_dict = defaultdict(lambda: self.lr)
        else:
            self.lr_dict = defaultdict(lambda: None)
            self.lr_dict.update(self.lr)

        # Same logic for weight decay
        # (but is always just set to 0.0 and hasn't been ablated)
        if self.wd is None or isinstance(self.wd, float):
            self.wd_dict = defaultdict(lambda: self.wd)
        else:
            self.wd_dict = defaultdict(lambda: None)
            self.wd_dict.update(self.wd)

    def forward(self, state=None, rate=1.0, L=None):
        """General interface to generate a global convolution kernel.

        state: Initial state for recurrent updates.
            E.g. for SSMs, this should have shape (B, H, N) (batch, d_model, d_state).
        rate: Relative sampling rate.
        L: Target kernel length.

        Returns:
          - (C, H, L) (channels, d_model, l_kernel) The convolution kernel.
          - (B, H, L) (batch, d_model, l_kernel)
              Extra information for how the state affects the output of convolving by kernel.
        """
        raise NotImplementedError

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

    def _setup_step(self, **kwargs):
        """Convert a model into a recurrent mode for autoregressive inference."""
        raise NotImplementedError

    def step(self, x, state, **kwargs):
        """Step the model for one timestep with input x and recurrent state."""
        raise NotImplementedError

    def default_state(self, *args, **kwargs):
        """Return a default initial state."""
        raise NotImplementedError

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through the kernel."""
        raise NotImplementedError

    @property
    def d_state(self):
        """Implement this for interfaces that want to interact with a stateful layer (i.e. SSMs).

        Currently the only codepath that might use this is the StateDecoder, which is not used.
        """
        raise NotImplementedError

    @property
    def state_to_tensor(self):
        """Same as d_state, only needed for niche codepaths involving recurrent state."""
        raise NotImplementedError

class SSMKernel(Kernel):
    """Parent class for different SSM parameterizations.

    This class is abstract and only defines some initializations and flags that are common to all SSM variants.
    It is instantiated by subclasses SSMKernel{Dense,Real,Diag,DPLR}.

    Options:
    d_state (N): State size (dimensionality of parameters A, B, C). Generally shouldn't need to be adjusted and doens't affect speed much for most kernels (e.g. S4, S4D).
    deterministic: Use a deterministic initialization for dt, A, B, C.
        Useful for debugging as well as constructing a simple exponential decay kernel (e.g. used in S4ND image->video inflation).

    dt_min, dt_max: min and max values for the step size dt
    dt_tie: Keep dt tied across the N dimensions of the state. Although this theoretically makes more sense, models such as S5 and Mega have found slightly improvements by setting it to False.
    dt_transform: Transform function for parameterization of dt (default 'softplus', used to be 'exp')

    rank: Rank of low-rank correction for DPLR mode. Needs to be increased for init "legt".
    n_ssm: Number of independent trainable (A, B) SSMs, e.g.
        `n_ssm=1` means all A/B parameters are tied across the H different instantiations of C.
        `n_ssm=None` means all H SSMs are completely independent.
        Generally, changing this option can save parameters but doesn't affect performance or speed much.
        This parameter must divide H.
    init: Options for initialization of (A, B). For DPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin).
    init_args: Extra arguments passed into initialization function (see dplr.py for options).
    """

    def init_dt(self):
        # Generate dt
        if self.deterministic:  # Meant for debugging
            assert self.dt_tie, "Deterministic dt initialization is tied"
            assert self.dt_transform == 'exp', "Deterministic dt transform should be 'exp' for simplicity"
            inv_dt = torch.exp(torch.linspace(math.log(self.dt_min), math.log(self.dt_max), self.H)).unsqueeze(-1) # (H 1)
        else:
            shape = (self.H, 1) if self.dt_tie else (self.H, self.N//2)
            # Initialize log dt
            inv_dt = torch.rand(*shape, dtype=self.dtype) * (
                math.log(self.dt_max) - math.log(self.dt_min)
            ) + math.log(self.dt_min)
            if self.dt_transform != 'exp':
                inv_dt = inv_transform(torch.exp(inv_dt), self.dt_transform)

        return inv_dt

    def init_ssm_real(self):
        """Returns (dense, real) (A, B, C) parameters for init options."""
        # Generate A, B
        A, B = hippo.transition(self.init, self.N)
        A = torch.as_tensor(A, dtype=self.dtype)
        B = torch.as_tensor(B, dtype=self.dtype)[:, 0]
        B = repeat(B, 'n -> v n', v=self.n_ssm).clone().contiguous()
        A = repeat(A, 'n m -> v n m', v=self.n_ssm).clone().contiguous()

        # Generate C
        if self.deterministic:
            C = torch.zeros(self.channels, self.H, self.N, dtype=self.dtype)
            C[..., :1] = 1.0
        else:
            C = torch.randn(self.channels, self.H, self.N, dtype=self.dtype)

        return A, B, C

    def __init__(
        self,
        # General Kernel arguments for parent class
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        # SSM arguments
        d_state: int = 64,
        deterministic: bool = False,
        # dt options
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_tie: bool = True,
        dt_transform: str = 'exp',
        # (A, B, C) options
        rank: int = 1,
        n_ssm: Optional[int] = None,
        measure: Optional[str] = None,
        init: Optional[str] = "legs",
        # Extra hyperparameters for initialization
        **init_args,
    ):
        super().__init__(d_model=d_model, channels=channels, l_max=l_max, lr=lr, wd=wd, verbose=verbose)
        self.N = d_state
        self.dtype, self.cdtype = torch.float, torch.cfloat
        self.deterministic = deterministic
        # dt options
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_tie = dt_tie
        self.dt_transform = dt_transform
        # SSM options (A, B, C)
        self.rank = rank
        self.n_ssm = n_ssm if n_ssm is not None else self.H
        if measure is not None:
            log.warning("Warning: 'measure' option changed to 'init' and will be removed in a future version.")
            assert init is None, "'measure' and 'init' cannot both be passed into SSMKernel"
            init, measure = measure, init
        self.init = init
        self.init_args = init_args

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        This is a generic version of this functionality that works for SSMs.
        It is currently used by SSMKernelDense and SSMKernelDPLR.
        This is a suboptimal implementation; it is recommended to use SSMKernelDiag
        if this functionality is desired.

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        # Construct dA, dB matrices
        dA, dB = self._setup_state() # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_state(self):
        """Register dA and dB to module."""
        raise NotImplementedError

    @property
    def d_state(self):
        """d_state and state_to_tensor are used by specific decoders.

        These were used in earlier versions and should not be needed in general.
        """
        return self.H * self.N

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class perturbation(nn.Module):
    def __init__(self,N,gamma):
        super().__init__()
        self.gamma = gamma
        self.N = N
        self.P = nn.Parameter(torch.view_as_real(torch.rand([N,N], dtype=torch.complex64) * gamma * 0.5))
       
        A = np.zeros([N,N], dtype='cfloat')
        for i in range(N):
            for j in range(i):
                A[i,j] = math.sqrt(2*i+1) * math.sqrt(2*j+1)
            A[i,i] = i+1
        self.A = torch.tensor(A, requires_grad=False)
       
    def forward(self):
        P = torch.view_as_complex(self.P)
        _, V = torch.linalg.eig(self.A + P)
        return torch.linalg.cond(V), torch.linalg.norm(P)


class perturbation_trainer():
    def __init__(self,N,lamb,lr=0.3,epochs=10000):
        self.N = N
        self.lamb = lamb
        self.lr = lr
        self.epochs = epochs

    def train_P(self):
        model = perturbation(self.N,0.2)
        opt = optim.Adam(model.parameters(),lr=self.lr,weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=100, gamma=0.98)
        for i in range(self.epochs):
            out,out2= model()
            loss = nn.functional.mse_loss(out, torch.tensor([0],dtype=torch.float64)) + self.lamb * out2
            if i % (self.epochs // 100) == 0:
                print('Epoch: ', i, ' Condition number: ', out.item(), ' Norm: ', out2.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
        return torch.view_as_complex(model.P).detach().numpy(), out.item(), out2.item()

class SSMKernelDiag(SSMKernel):
    """SSM kernel using diagonal state matrix (S4D model).

    Options:
    disc: ['zoh' | 'bilinear' | 'dss'] Discretization options.
    dt_fast:  (experimental) Parameterize inv_dt under sinh function.
        (Ohno et al. "Fast Saturating Gate for Learning Long Time Scales with RNNs")
    real_transform, imag_transform: ['none' | 'exp' | 'relu' | 'sigmoid' | 'softplus']
        Parameterize the real/imag parts of the diagonal of A under this function.
    bandlimit: Mask high frequencies of the kernel (indices corresponding to
        diagonal elements with large imaginary part). Introduced in S4ND paper.
    kernel: ['cuda' | 'keops' | 'naive'] Options for Vandermonde/Cauchy kernel (in order of efficiency).
    force_real : Force A to have 0 imaginary part, to emulate EMA.
    """

    def __init__(
        self,
        disc: str = 'zoh',  # Change to 'bilinear' to match S4, but should make little difference either way
        dt_fast: bool = False,
        real_transform: str = 'exp',
        imag_transform: str = 'none',
        bandlimit: Optional[float] = None,
        backend: str = 'cuda',
        force_real: bool = False,
        scale_factor: float = 2,
        decay_exp: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.disc = disc
        self.dt_fast = dt_fast
        self.real_transform = real_transform
        self.imag_transform = imag_transform
        self.bandlimit = bandlimit
        self.backend = backend
        self.force_real = force_real
        self.scale_factor = scale_factor
        self.decay_exp = decay_exp

        # Initialize dt, A, B, C
        inv_dt = self.init_dt()

        H = self.n_ssm
        Nh = self.N

        Han = torch.randn(H, Nh, dtype=torch.cfloat) / math.sqrt(Nh) / self.scale_factor

        # The DPLR case subclasses this and uses P
        self.register_params(Han, inv_dt)

    def register_params(self, Han, inv_dt):
        """Process the initialization into form of trainable parameters.

        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        assert self.backend in ['cuda', 'keops', 'naive']

        if self.dt_fast: inv_dt = torch.asinh(inv_dt)

        self.repeat = self.H // Han.size(0)

        # Register dt, B, A
        self.register("inv_dt", inv_dt, self.lr_dict['dt'], self.wd_dict['dt'])
        self.register("Han", _c2r(Han), self.wd_dict['A'], self.wd_dict['A'])

    def _get_params(self, rate=1.0):
        """Process the internal parameters."""

        if self.dt_fast: inv_dt = torch.sinh(self.inv_dt)
        else: inv_dt = self.inv_dt
        dt = param_transform(inv_dt, self.dt_transform) * rate # (H N)

        return dt

    def forward(self, L, state=None, rate=1.0):
        """See Kernel.forward() for argument documentation."""

        dt_expanded = self._get_params(rate)

        Hankel = torch.view_as_complex(self.Han)
        Hankel = Hankel * ((torch.arange(self.N, device=dt_expanded.device)+1) ** self.decay_exp)

        samplers = torch.exp(2j * math.pi * repeat(torch.arange(L, device=dt_expanded.device) / L, 'n -> h n', h=self.H))
        samplers_dt = ((1 + dt_expanded) * samplers + dt_expanded - 1) / ((-1 + dt_expanded) * samplers + dt_expanded + 1)
        samplers_angle = torch.angle(samplers_dt)

        enumer = -torch.arange(self.N, dtype=samplers_angle.dtype, device=samplers_angle.device) - 1  # Ensure K is the same dtype and device as H
        samplers_angle = samplers_angle.unsqueeze(-1)
        samplers_angle = torch.exp(1j * samplers_angle * enumer) # (H L N)
        K = torch.einsum('hn, hln -> hl', Hankel, samplers_angle)
        K = K.unsqueeze(0)

        return K, 0

    def _setup_step(self):
        """Set up dA, dB, dC discretized parameters for stepping."""

        dt, A, B, C, = self._get_params()
        # Incorporate dt into A
        dtA = dt * A  # (H N)

        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
        self.dC = C

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = 0
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        """Pass the state forward through an entire sequence."""
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        # Dispatch which Vandermonde kernel to use
        if has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde_transpose = log_vandermonde_transpose_keops
        else:
            log_vandermonde_transpose = log_vandermonde_transpose_naive
        v = log_vandermonde_transpose(u, 0, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state

kernel_registry = {
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
}

class FFTConv(SequenceModule):
    """Implements an FFT Convolution around a convolution kernel.

    d_model (H): Model dimension (in CNN terminology, this would be "channels").
    l_max (L): The maximum kernel length. Set l_max=None to always use a global kernel.
    channels: Can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this; instead, increase d_model for larger models.
    bidirectional: If True, convolution kernel will be two-sided.
    activation: Activation after the full convolution.
    transposed, dropout, tie_dropout: More general model options, see SequenceModule.
    mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D. Other options can be found in the kernel registry.

    kernel_args: See the class .kernel.SSMKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode", "init", "dt_min", "dt_max", "lr"
    """

    def __init__(
        self,
        d_model,
        l_max=None,
        channels=1,
        swap_channels=False,
        bidirectional=False,
        activation='gelu', # Activation after layer
        transposed=True,
        dropout=0.0,
        tie_dropout=False,
        drop_kernel=0.0,
        mode='dplr',
        kernel=None,
        **kernel_args,  # Arguments passed into inner convolution kernel
    ):
        super().__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.swap_channels = swap_channels


        if activation is not None and activation.startswith('glu'):
            channels *= 2
        self.activation = Activation(activation, dim=1 if self.transposed else -1)

        self.D = nn.Parameter(torch.randn(channels, self.d_model))

        if self.bidirectional:
            channels *= 2

        # Inner convolution kernel
        if mode is not None:
            assert kernel is None, "Pass either mode or kernel but not both"
            # log.info(
            #     "Argument 'mode' is deprecated and renamed to 'kernel',"
            #     "and will be removed in a future version."
            # )
            kernel, mode = mode, kernel
        kernel_cls = kernel_registry[kernel]
        self.kernel = kernel_cls(
            d_model=self.d_model,
            l_max=self.l_max,
            channels=channels,
            **kernel_args,
        )

        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

    def forward(self, x, state=None, rate=1.0, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B D L) if self.transposed else (B L D)
        """

        # Always work with (B D L) dimension in this module
        if not self.transposed: x = x.transpose(-1, -2)
        L = x.size(-1)

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state =  self.kernel(L=l_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        x_f = torch.fft.fft(x) # (B H L)
        y_f = contract('bhl,chl->bchl', x_f, k)
        y = torch.fft.ifft(y_f).real # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', x, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None


        # Reshape to flatten channels
        if self.swap_channels:
            y = rearrange(y, 'b c h l -> b (h c) l')
        else:
            y = rearrange(y, 'b c h l -> b (c h) l')

        y = self.drop(y)  # DropoutNd better with transposed=True

        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)

        return y, next_state


    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        y, next_state = self.kernel.step(x, state) # (B C H)
        y = y + x.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.kernel.d_state

    @property
    def d_output(self):
        return self.d_model * self.channels

    @property
    def state_to_tensor(self):
        return self.kernel.state_to_tensor

class S4Block(SequenceModule):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    transposed=False,
                    initializer=initializer,
                    activation=None,
                    activate=False,
                    weight_norm=weight_norm,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        self.layer = FFTConv(d_model, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)

        # Pointwise operations

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        '''
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.layer.d_output,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )
        '''
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.d_model, 2*self.d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )



    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(x, **kwargs)


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)

        y = rearrange(y, 'b d ... -> b ... d')
        y = self.output_linear(y)
        y = rearrange(y, 'b d ... -> b ... d')

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        # y = self.mult_activation(y)
        y = nn.GELU(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor
