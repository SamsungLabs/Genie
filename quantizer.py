import logging
import torch
from torch import nn

log = logging.getLogger(__name__)

def uniform_quantize(x, delta, zero_point, n_bits):
    x_int = torch.round(x / delta)
    x_q = torch.clamp(x_int + zero_point, 0, 2**n_bits - 1)
    x_deq = (x_q-zero_point) * delta
    return x_deq
    
def init_scale(x, n_bits, symmetric, channel_wise, signed=True):
    # parallel batch
    n_batch = x.shape[0] if channel_wise else 1
    x_flat = x.reshape(n_batch, -1).detach()

    best_score = torch.full([n_batch], 1e+10, device=x.device)

    # Four cases need to be considered: {signed, unsigned} x {symmetric, asymmetric}
    if symmetric:
        max_value = x_flat.abs().max(dim=1).values
        x_max = max_value
        x_min = -max_value if signed else torch.zeros_like(x_max)
    else:
        x_max = x_flat.max(dim=1).values
        x_min = x_flat.min(dim=1).values if signed else torch.max(x_flat.min(dim=1).values, torch.tensor([0.]))

    delta = torch.zeros_like(best_score)
    zero_point = torch.zeros_like(best_score)

    # Finding scales
    for clip_ratio in torch.arange(1.0, 0.0, -0.01):
        new_max, new_min = x_max * clip_ratio, x_min * clip_ratio

        new_delta = (new_max-new_min) / (2**n_bits - 1)
        new_zeropoint = (- new_min/new_delta).round()
        x_q = uniform_quantize(x_flat, new_delta.unsqueeze(1), new_zeropoint.unsqueeze(1), n_bits)
        score = (x_flat-x_q).abs().pow(2.4).mean(dim=1)

        delta = torch.where(score < best_score, new_delta, delta)
        zero_point = torch.where(score < best_score, new_zeropoint, zero_point)
        best_score = torch.minimum(score, best_score)

    if torch.any(delta < 1e-10):
        log.warning(f'Quantization range close to zero: [{delta}]')

    target_dim = [-1, *[1]*(len(x.shape)-1)]
    return delta.view(target_dim), zero_point.view(target_dim)


class RectifiedSigmoid(nn.Module):
    def __init__(self, gamma, zeta):
        super(RectifiedSigmoid, self).__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(torch.sigmoid(x)*(self.zeta-self.gamma) + self.gamma, 0, 1)

    def inverse(self, y):
        """return x that satisfies y = RectifiedSigmoid(x)"""
        return -torch.log((self.zeta-self.gamma)/(y-self.gamma) - 1)

class WeightQuantizer(nn.Module):
    """
    Implementation of AdaRound and Genie quantizer
    References:
    https://arxiv.org/abs/2212.04780
    https://arxiv.org/abs/2004.10568
    """

    def __init__(self, weight: torch.Tensor, n_bits):
        super(WeightQuantizer, self).__init__()

        self.n_bits = n_bits
        self.scale, self.zero_point = init_scale(
            weight, n_bits=self.n_bits, symmetric=False, channel_wise=True, signed=True)
        self.scale = nn.Parameter(self.scale)
        self.x_floor = (weight.detach() / self.scale).floor().detach()

        # initialize sigmoid
        self.sigmoid = RectifiedSigmoid(-0.1, 1.1)
        residue = weight/self.scale - torch.floor(weight/self.scale)
        self.bit_logit = nn.Parameter(self.sigmoid.inverse(residue))

        # rounding mode selection (default: hard-rounding)
        self.train_mode = False

    def forward(self):
        x_int = self.x_floor + (self.soft_target() if self.train_mode else (self.bit_logit >= 0).float())
        x_quant = torch.clamp(x_int + self.zero_point, 0, 2**self.n_bits - 1)
        x_deq = (x_quant-self.zero_point) * self.scale
        return x_deq

    def extra_repr(self) -> str:
        return f'n_bits={self.n_bits}'

    def soft_target(self):
        return self.sigmoid(self.bit_logit)

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


class ActivationQuantizer(nn.Module):
    """
    An implementation of the Learned Step Size Quantization and QDrop.
    References:
    https://arxiv.org/pdf/1902.08153.pdf
    https://arxiv.org/abs/2203.05740.pdf
    """

    def __init__(self, n_bits):
        super(ActivationQuantizer, self).__init__()
        self.n_bits = n_bits
        self.scale = None
        self.initialized = False
        self.signed = None
        self.train_mode = False

    def forward(self, x):
        if not self.initialized:
            self.signed = x.min() < 0
            self.scale = nn.Parameter(
                init_scale(x, n_bits=self.n_bits, symmetric=True, channel_wise=False, signed=self.signed)[0])
            self.initialized = True

        Qn = - 2**(self.n_bits-1) if self.signed else 0
        Qp = 2**(self.n_bits-1) - 1 if self.signed else 2**self.n_bits - 1

        v, s = x, self.scale
        v_bar = round_ste(torch.clamp(v / s, Qn, Qp))
        v_hat = v_bar * s

        # use QDrop
        if self.train_mode:
            return torch.where(torch.rand_like(x) < 0.5, v_hat, x)
        else:
            return v_hat

    def extra_repr(self) -> str:
        return f'n_bits={self.n_bits}, signed={self.signed}, scale={self.scale}'