"""Helper functions for Notebook 3 (Normalizing Flows).

Contains the rational quadratic spline of Durkan et al. 2019 ("Neural Spline Flows",
https://arxiv.org/abs/1906.04032), which is used by the spline coupling layer in the notebook.
"""
import numpy as np
import torch
import torch.nn.functional as F


def spline_knots(unnormalized_widths, unnormalized_heights, bound=3.0, min_bin_size=1e-3):
    """Turn unconstrained network outputs into the bins and knots of a spline on [-bound, bound].

    unnormalized_widths, unnormalized_heights: [..., K] (K bins)
    Returns the bin widths and heights [..., K] and the knot positions knots_x, knots_y [..., K+1].
    """
    K = unnormalized_widths.shape[-1]

    # Softmax to get widths/heights which are positive and sum up to one.
    # (The minimal bin size avoids numerical problems with extremely narrow bins.)
    w = min_bin_size + (1 - min_bin_size * K) * torch.softmax(unnormalized_widths, dim=-1)
    h = min_bin_size + (1 - min_bin_size * K) * torch.softmax(unnormalized_heights, dim=-1)

    # Cumulative widths/heights give the K+1 knots, rescaled from [0, 1] to [-bound, bound]
    knots_x = 2 * bound * F.pad(torch.cumsum(w, dim=-1), (1, 0), value=0.0) - bound
    knots_y = 2 * bound * F.pad(torch.cumsum(h, dim=-1), (1, 0), value=0.0) - bound

    return 2 * bound * w, 2 * bound * h, knots_x, knots_y


def spline_derivatives(unnormalized_derivatives, min_derivative=1e-3):
    """Derivatives at the K+1 knots from the unconstrained network outputs [..., K-1].

    The derivatives have to be positive (softplus). The offset makes the derivative 1 for an input of 0.
    At the two boundary knots the derivative is fixed to 1 to match the linear tails.
    """
    d = min_derivative + F.softplus(unnormalized_derivatives + np.log(np.e - 1))
    return F.pad(d, (1, 1), value=1.0)


def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                              inverse=False, bound=3.0, min_bin_size=1e-3, min_derivative=1e-3):
    """Monotonic rational quadratic spline, applied elementwise.

    inputs: [...]
    unnormalized_widths, unnormalized_heights: [..., K] (K bins)
    unnormalized_derivatives: [..., K-1] (derivatives at the K-1 inner knots)
    inverse: if True, apply the inverse transformation

    Returns the outputs [...] and log|d output / d input| [...].
    """
    # Outside of [-bound, bound] we use the identity (linear tails)
    inside = (inputs >= -bound) & (inputs <= bound)
    x = inputs.clamp(-bound, bound)

    # 1. Bins, knots and derivatives at the knots
    w, h, knots_x, knots_y = spline_knots(unnormalized_widths, unnormalized_heights, bound, min_bin_size)
    d = spline_derivatives(unnormalized_derivatives, min_derivative)

    # 2. Find the bin of each input (search in y for the inverse) and gather its parameters
    knots = knots_y if inverse else knots_x
    bin_idx = torch.searchsorted(knots[..., 1:-1].contiguous(), x[..., None])

    def gather(params):
        return torch.gather(params, -1, bin_idx).squeeze(-1)

    x_k, y_k = gather(knots_x[..., :-1]), gather(knots_y[..., :-1])  # lower knot of the bin
    w_k, h_k = gather(w), gather(h)                                  # width and height of the bin
    d_k, d_k1 = gather(d[..., :-1]), gather(d[..., 1:])              # derivatives at lower and upper knot
    s_k = h_k / w_k                                                  # slope of the bin

    # 3. Evaluate the spline
    if inverse:
        # y -> x: solve a quadratic equation for the relative position xi in the bin
        y = x
        a = h_k * (s_k - d_k) + (y - y_k) * (d_k1 + d_k - 2 * s_k)
        b = h_k * d_k - (y - y_k) * (d_k1 + d_k - 2 * s_k)
        c = -s_k * (y - y_k)
        xi = 2 * c / (-b - torch.sqrt(b**2 - 4 * a * c))
        denominator = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
        outputs = x_k + xi * w_k
    else:
        # x -> y: relative position xi in the bin, then the formula of the spline
        xi = (x - x_k) / w_k
        denominator = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
        outputs = y_k + h_k * (s_k * xi**2 + d_k * xi * (1 - xi)) / denominator

    # 4. Jacobian: dy/dx (in closed form)
    log_det = (2 * torch.log(s_k)
               + torch.log(d_k1 * xi**2 + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi)**2)
               - 2 * torch.log(denominator))
    if inverse:
        log_det = -log_det

    # Identity outside of the interval
    outputs = torch.where(inside, outputs, inputs)
    log_det = torch.where(inside, log_det, torch.zeros_like(log_det))

    return outputs, log_det


# ------------------------------------------------------------------------------------------------------
# Conditional RealNVP: the conditional flow from the end of Notebook 3, for any number of dimensions.
# Notebook 4 imports it from here to build NPE and NLE from scratch.
# ------------------------------------------------------------------------------------------------------

class ConditionalCouplingLayer(torch.nn.Module):
    """Affine coupling layer whose scale and translation depend on the masked input and on a context."""

    def __init__(self, input_dim, context_dim, hidden_dim, mask, max_log_scale=1.0):
        super().__init__()
        self.register_buffer('mask', mask)
        # Bound of the log-scale s of one layer: |s| < max_log_scale (Notebook 3 used 1)
        self.max_log_scale = max_log_scale
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + context_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, input_dim * 2)
        )

    def scale_and_translation(self, x, context):
        # s and t depend on the masked input AND on the context
        st = self.net(torch.cat([x * self.mask, context], dim=1))
        s, t = st.chunk(2, dim=1)
        s = self.max_log_scale * torch.tanh(s / self.max_log_scale) * (1 - self.mask)
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, context):
        s, t = self.scale_and_translation(x, context)
        return x * torch.exp(s) + t, torch.sum(s, dim=1)

    def inverse(self, y, context):
        s, t = self.scale_and_translation(y, context)
        return (y - t) * torch.exp(-s)


class ConditionalRealNVP(torch.nn.Module):
    """Conditional normalizing flow q(x | context) with a standard normal base distribution.

    forward:  x -> z (normalizing direction), returns z and log|det dz/dx|
    inverse:  z -> x (generative direction)
    log_prob: log q(x | context)
    sample:   one sample x for every row of the context
    """

    def __init__(self, input_dim, context_dim, hidden_dim=64, n_layers=6, max_log_scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList()
        # Alternating masks: [1, 0, 1, 0, ...] and [0, 1, 0, 1, ...]
        mask = (torch.arange(input_dim) % 2 == 0).float()
        for _ in range(n_layers):
            self.layers.append(ConditionalCouplingLayer(input_dim, context_dim, hidden_dim, mask, max_log_scale))
            mask = 1 - mask

    def forward(self, x, context):
        log_det_sum = 0
        for layer in self.layers:
            x, log_det = layer(x, context)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum

    def inverse(self, z, context):
        for layer in reversed(self.layers):
            z = layer.inverse(z, context)
        return z

    def log_prob(self, x, context):
        # Change of variables: log q(x) = log N(z | 0, 1) + log|det dz/dx|
        z, log_det = self.forward(x, context)
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob_z + log_det

    def sample(self, context):
        z = torch.randn(len(context), self.input_dim)
        return self.inverse(z, context)
