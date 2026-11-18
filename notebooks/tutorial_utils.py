"""Shared toy problems for the SBI notebooks (Notebook 4 and later).

Every notebook solves one of two problems with a different tool. To make the results comparable, the
simulators, the fixed benchmark simulations, the exact posteriors and some plotting functions live here.

Problem A, "Gaussian toy" (Notebooks 4, 5, 6, 9):
    theta = (mean, log_std), data x = 10 draws from N(mean, std^2), flat prior. The exact posterior is known.

Problem B, "log-normal toy" (Notebooks 7, 8):
    3 parameters, 30-dimensional data vector with skewed noise, training simulations which do not follow the prior.
"""
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt


# ======================================================================================================
# Problem A: Gaussian toy model
# ======================================================================================================

GAUSSIAN_PARAM_NAMES = ['Mean', 'Log Std']
# Flat prior: mean in [-5, 5], log_std in [-2, 1]
GAUSSIAN_PRIOR_LOW = torch.tensor([-5.0, -2.0])
GAUSSIAN_PRIOR_HIGH = torch.tensor([5.0, 1.0])


def gaussian_simulator(theta):
    """Simulator of the Gaussian toy model.

    theta: parameters (mean, log_std), shape [N, 2] (or [2] for a single simulation)
    Returns x: ten draws from a Gaussian with this mean and standard deviation, shape [N, 10]
    """
    theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, 2)
    mean = theta[:, :1]
    std = torch.exp(theta[:, 1:])
    return mean + std * torch.randn(len(theta), 10)


def sample_gaussian_prior(num_samples):
    """Draw parameters from the flat prior, shape [num_samples, 2]."""
    return GAUSSIAN_PRIOR_LOW + (GAUSSIAN_PRIOR_HIGH - GAUSSIAN_PRIOR_LOW) * torch.rand(num_samples, 2)


def gaussian_benchmark(num_train=10000, num_test=500, seed=0):
    """Fixed set of simulations, identical in every notebook (independent of the global random seed).

    Returns an object with the attributes
        theta_train, x_train: training simulations, parameters drawn from the prior
        theta_test, x_test:   test simulations for coverage tests, parameters drawn from the prior
        theta_obs, x_obs:     the "observation" [1, 10] and the parameters [2] it was simulated with
    """
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    theta_train = sample_gaussian_prior(num_train)
    x_train = gaussian_simulator(theta_train)
    theta_test = sample_gaussian_prior(num_test)
    x_test = gaussian_simulator(theta_test)
    theta_obs = torch.tensor([2.0, -0.5])     # mean = 2, std = exp(-0.5) = 0.6
    x_obs = gaussian_simulator(theta_obs)
    torch.set_rng_state(rng_state)
    return SimpleNamespace(theta_train=theta_train, x_train=x_train, theta_test=theta_test, x_test=x_test,
                           theta_obs=theta_obs, x_obs=x_obs)


def analytic_posterior(x_obs, n_grid=(1000, 300)):
    """Exact posterior p(mean, log_std | x_obs) of the Gaussian toy model on a grid.

    Returns the grid of the mean [n_grid[0]], the grid of log_std [n_grid[1]] and the normalized
    posterior density [n_grid[0], n_grid[1]].
    """
    mean_grid = torch.linspace(GAUSSIAN_PRIOR_LOW[0], GAUSSIAN_PRIOR_HIGH[0], n_grid[0])
    log_std_grid = torch.linspace(GAUSSIAN_PRIOR_LOW[1], GAUSSIAN_PRIOR_HIGH[1], n_grid[1])
    mean, log_std = torch.meshgrid(mean_grid, log_std_grid, indexing='ij')
    x = torch.as_tensor(x_obs, dtype=torch.float32).reshape(-1)

    # Log-likelihood of all data points: sum_i log N(x_i | mean, std^2)
    log_likelihood = -len(x) * log_std - ((x - mean[..., None])**2).sum(dim=-1) / (2 * torch.exp(2 * log_std))

    # The prior is flat, so the posterior is the likelihood, normalized on the grid
    posterior_grid = torch.exp(log_likelihood - log_likelihood.max())
    cell_area = (mean_grid[1] - mean_grid[0]) * (log_std_grid[1] - log_std_grid[0])
    posterior_grid = posterior_grid / (posterior_grid.sum() * cell_area)
    return mean_grid, log_std_grid, posterior_grid


def analytic_moments(x_obs):
    """Mean [2] and standard deviation [2] of the exact posterior of the Gaussian toy model."""
    mean_grid, log_std_grid, posterior_grid = analytic_posterior(x_obs)
    means, stds = [], []
    for grid, marginal in ((mean_grid, posterior_grid.sum(dim=1)), (log_std_grid, posterior_grid.sum(dim=0))):
        marginal = marginal / marginal.sum()
        means.append((grid * marginal).sum())
        stds.append(((grid - means[-1])**2 * marginal).sum().sqrt())
    return torch.stack(means), torch.stack(stds)


def plot_posterior_2d(ax, samples, x_obs, theta_true=None, title='', label='Samples'):
    """Scatter plot of posterior samples [N, 2] with the 68% and 95% credible regions of the exact posterior."""
    samples = np.asarray(samples)
    points = ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.2)
    ax.scatter([], [], s=10, color=points.get_facecolor()[0][:3], label=label)   # legend entry

    mean_grid, log_std_grid, posterior_grid = analytic_posterior(x_obs)
    # Density levels which enclose 95% and 68% of the probability
    sorted_density = posterior_grid.flatten().sort(descending=True).values
    enclosed_probability = sorted_density.cumsum(dim=0) / sorted_density.sum()
    levels = [sorted_density[torch.searchsorted(enclosed_probability, p)].item() for p in (0.95, 0.68)]
    ax.contour(mean_grid, log_std_grid, posterior_grid.T, levels=levels, colors='black')
    ax.plot([], [], color='black', label='Exact posterior (68%, 95%)')

    if theta_true is not None:
        ax.plot(theta_true[0], theta_true[1], '*', color='red', markersize=12, label='True parameters')
    exact_mean, exact_std = analytic_moments(x_obs)
    ax.set(xlim=(exact_mean[0] - 6 * exact_std[0], exact_mean[0] + 6 * exact_std[0]),
           ylim=(exact_mean[1] - 6 * exact_std[1], exact_mean[1] + 6 * exact_std[1]),
           xlabel=GAUSSIAN_PARAM_NAMES[0], ylabel=GAUSSIAN_PARAM_NAMES[1], title=title)


def plot_posterior(samples_dict, x_obs, theta_true):
    """2D posterior and the two marginal distributions for one or several sets of samples.

    samples_dict: {name: samples [N, 2]}
    """
    mean_grid, log_std_grid, posterior_grid = analytic_posterior(x_obs)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))

    for name, samples in samples_dict.items():
        plot_posterior_2d(axs[0], samples, x_obs, label=name)
    # Show the legend entry for the exact posterior only once
    handles, labels = axs[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    star, = axs[0].plot(theta_true[0], theta_true[1], '*', color='red', markersize=12)
    unique['True parameters'] = star
    axs[0].legend(unique.values(), unique.keys(), loc='upper left')

    # Marginal distributions: integrate the exact posterior over the other parameter
    marginals = (posterior_grid.sum(dim=1) * (log_std_grid[1] - log_std_grid[0]),
                 posterior_grid.sum(dim=0) * (mean_grid[1] - mean_grid[0]))
    for i, (grid, marginal) in enumerate(zip((mean_grid, log_std_grid), marginals)):
        limits = axs[0].get_xlim() if i == 0 else axs[0].get_ylim()
        for name, samples in samples_dict.items():
            axs[i + 1].hist(np.asarray(samples)[:, i], bins=50, range=limits, density=True, histtype='step', lw=1.5, label=name)
        axs[i + 1].plot(grid, marginal, color='black', label='Exact posterior')
        axs[i + 1].axvline(theta_true[i], color='red', ls='--', label='True parameter')
        axs[i + 1].set(xlim=limits, xlabel=GAUSSIAN_PARAM_NAMES[i])
    axs[1].legend()
    plt.show()


# ======================================================================================================
# Problem B: log-normal toy model
# ======================================================================================================

LOGNORMAL_PARAM_NAMES = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
# Flat prior: all three parameters in [-2, 2]
LOGNORMAL_PRIOR_LOW = -2 * torch.ones(3)
LOGNORMAL_PRIOR_HIGH = 2 * torch.ones(3)


def model_fn(theta):
    """Noise-free data vector [N, 30] for parameters theta [N, 3] (numpy).

    Three blocks of ten entries, proportional to theta_1, theta_2^2 and theta_3^3.
    """
    theta_pow = (theta**np.arange(1, 4)[None, :])
    theta_mult = np.ones((theta.shape[0], 3, 10)) * np.arange(1, 31).reshape(3, 10)[np.newaxis, ...]
    return (theta_pow[:, :, np.newaxis] * theta_mult).reshape(-1, 30)


# Covariance of the Gaussian variable g behind the noise (neighbouring entries are slightly correlated)
cov = np.eye(30)
for i in range(29):
    cov[i, i+1] = 0.02
    cov[i+1, i] = 0.02
cov_inv = np.linalg.inv(cov)

# Mean of the log-normal noise exp(g) with g ~ N(1, 1): exp(mu + sigma^2 / 2)
noise_mean = np.exp(1.5)


def generate_data(n_samples=1000, theta=None, grid='wide'):
    """Simulator of the log-normal toy model.

    theta: parameters [n_samples, 3] (numpy). If None, they are drawn from one of three distributions:
        grid='wide':     uniform in [-2, 2]^3 (this is the prior)
        grid='narrow':   uniform in [-1, 1]^3
        grid='combined': half of the samples 'narrow', half 'wide'
    Returns the data vectors x [n_samples, 30] and the parameters theta [n_samples, 3] as torch tensors.
    """
    if theta is None:

        if grid == 'wide':
            theta = np.random.uniform(-2, 2, (n_samples, 3))
        elif grid == 'narrow':
            theta = np.random.uniform(-1, 1, (n_samples, 3))
        elif grid == 'combined':
            theta_narrow = np.random.uniform(-1, 1, (int(n_samples//2), 3))
            theta_wide = np.random.uniform(-2, 2, (int(n_samples//2), 3))
            theta = np.concatenate([theta_narrow, theta_wide], axis=0)
        else:
            raise ValueError(f"Unsupported grid: {grid}")

    x = model_fn(theta)
    # Log-normal noise, shifted to zero mean
    noise = np.exp(np.random.multivariate_normal(np.ones(30), cov, theta.shape[0])) - noise_mean
    return torch.tensor(x + noise).float(), torch.tensor(theta).float()


def lognormal_benchmark(seed=42):
    """Fixed set of simulations, identical in every notebook (independent of the global random seed).

    Returns an object with the attributes (x: data vectors [N, 30], theta: parameters [N, 3])
        x_train_npe, theta_train_npe: 10000 simulations for the compression network, 'combined' (NOT the prior)
        x_val_npe, theta_val_npe:     1000 validation simulations, 'combined'
        x_train_nle, theta_train_nle: 5000 simulations for the likelihood estimator, 'combined'
        x_val_nle, theta_val_nle:     1000 validation simulations, 'combined'
        x_test, theta_test:           1000 test simulations, parameters drawn from the prior
        x_fiducial, theta_fiducial:   100 simulations at the fiducial model theta = (0, 0, 0)
        x_obs, theta_obs:             the "observation" [30] (one fiducial realization) and its parameters [3]
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    b = SimpleNamespace()
    b.x_train_npe, b.theta_train_npe = generate_data(10000, grid='combined')
    b.x_val_npe, b.theta_val_npe = generate_data(1000, grid='combined')
    b.x_train_nle, b.theta_train_nle = generate_data(5000, grid='combined')
    b.x_val_nle, b.theta_val_nle = generate_data(1000, grid='combined')
    b.x_test, b.theta_test = generate_data(1000, grid='wide')
    b.x_fiducial, b.theta_fiducial = generate_data(100, np.zeros((100, 3)))
    # The observation is a single realization at the fiducial model. We do NOT use the mean of the fiducial
    # realizations: because the noise is strongly skewed, such an averaged ("noise-free") data vector does not
    # look like a typical realization of the simulator, and analysing it with a likelihood (or posterior)
    # learned for single noisy realizations gives a biased result. (For Gaussian noise the mean would be fine.)
    b.x_obs, b.theta_obs = b.x_fiducial[0], b.theta_fiducial[0]
    np.random.set_state(rng_state)
    return b


def exact_log_posterior(theta, x):
    """Exact log-posterior (up to a constant) of the log-normal toy model.

    theta: parameters [n, 3] (numpy), x: one data vector [30] (numpy). Returns [n].
    """
    # The simulator adds the noise exp(g) - noise_mean with g ~ N(1, cov). Solve for exp(g):
    exp_g = x[None, :] - model_fn(theta) + noise_mean
    # exp(g) has to be positive, and theta has to be inside of the prior range
    allowed = np.all(exp_g > 0, axis=1) & np.all(np.abs(theta) <= 2, axis=1)
    g = np.log(np.where(exp_g > 0, exp_g, 1.0))
    # Gaussian log-probability of g, plus the Jacobian of the transformation from x to g: log|dg/dx| = -g
    log_prob = -0.5 * np.einsum('ni,ij,nj->n', g - 1, cov_inv, g - 1) - g.sum(axis=1)
    return np.where(allowed, log_prob, -np.inf)
