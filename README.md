# Machine Learning & Simulation-Based Inference for Astrophysics

This repository contains a collection of tutorial notebooks designed to introduce **Machine Learning (ML)** and **Simulation-Based Inference (SBI)** to Astrophysics students.

The goal is to build a strong foundational understanding by implementing key concepts from scratch (like Neural Networks and Normalizing Flows) before moving to implementations using libraries.

## Content

The tutorials are organized into nine notebooks. Notebooks 1 to 4 build everything from scratch, Notebooks 5 to 8 introduce three packages (and double as quickstart guides for them), and Notebook 9 compares all approaches:

### 1. `01_mlp_from_scratch.ipynb` (NumPy)
*   **Topic:** Building a Multi-Layer Perceptron (MLP) for regression without deep learning frameworks.
*   **Application:** Estimating galaxy distances from redshift (toy model).
*   **Concepts:**
    *   Forward and Backward propagation equations.
    *   Activation functions (ReLU, LeakyReLU, Sigmoid).
    *   Loss functions (MSE) and Regularization (L1/L2).
    *   Optimizers implementation (SGD, Momentum, Adam) and their comparison.
    *   Numerical gradient check of the backpropagation.
    *   Training vs. validation data, overfitting.

### 2. `02_torch_regression.ipynb` (PyTorch)
*   **Topic:** Introduction to PyTorch for building modular neural networks.
*   **Application:** Redshift-Distance regression (revisited).
*   **Concepts:**
    *   PyTorch Tensors and Autograd.
    *   `nn.Module` class structure.
    *   `DataLoader` and efficient batching.
    *   Training loops with `torch.optim`.
    *   Predicting a distribution (mean and variance) with a Gaussian likelihood loss.

### 3. `03_normalizing_flows.ipynb` (Flows from Scratch)
*   **Topic:** Density estimation using Normalizing Flows.
*   **Application:** Modeling the "Two Moons" dataset.
*   **Concepts:**
    *   Change of variables formula and Jacobian determinant.
    *   **RealNVP** (Real Non-Volume Preserving) Coupling Layers.
    *   **MAF** (Masked Autoregressive Flow) using MADE (Masked Autoencoder).
    *   **NSF** (Neural Spline Flow) with Rational Quadratic Spline coupling layers.
    *   Testing a flow (inverse, Jacobian, normalization).
    *   Conditional flows as the building block of SBI.
    *   Implemented from scratch using `torch.nn` primitives (the spline function itself lives in `notebooks/flow_utils.py`).

### 4. `04_sbi_from_scratch.ipynb` (SBI from Scratch)
*   **Topic:** Likelihood-free inference for simulations, implemented by hand.
*   **Application:** Parameter inference on a Gaussian toy model with a known exact posterior.
*   **Concepts:**
    *   **NPE** (Neural Posterior Estimation) and **NLE** (Neural Likelihood Estimation) with the conditional flow from Notebook 3.
    *   A simple MCMC sampler (random-walk Metropolis).
    *   The **TARP** coverage test, written by hand.

### 5. `05_sbi_inference.ipynb` (`sbi` Package)
*   **Topic:** The same problem with the [`sbi`](https://sbi.readthedocs.io/en/stable/) package; also a quickstart for it.
*   **Concepts:**
    *   The `sbi` workflow for NPE and NLE, amortization, choosing the density estimator, configuring the MCMC.
    *   Effect of the simulation budget.
    *   Built-in diagnostics (TARP, SBC) and how to read coverage plots.

### 6. `06_ltu_ili.ipynb` (`ltu-ili` Pipeline)
*   **Topic:** The same problem with [`ltu-ili`](https://github.com/maho3/ltu-ili); also a quickstart for it.
*   **Concepts:**
    *   Data loaders, priors, network ensembles, `InferenceRunner`.
    *   Validation metrics (single posteriors, coverage, rank histograms, TARP).
    *   Switching backends and running an analysis from configuration files.

### 7. `07_sbi_advanced.ipynb` (Compression + NLE by Hand)
*   **Topic:** Advanced SBI methods for complex simulations.
*   **Application:** Parameter inference on a Non-Gaussian toy model with a large data vector.
*   **Concepts:**
    *   Custom Density Estimators and Custom Training Loops (on `sbi` building blocks).
    *   Learned data compression (embedding network trained with NPE, VMIM) followed by NLE.
    *   Why NPE fails if the training simulations do not follow the prior.
    *   MCMC sampling with [`emcee`](https://emcee.readthedocs.io), a batched ensemble sampler, and coverage tests with **TARP** ([`tarp`](https://github.com/Ciela-Institute/tarp)).
    *   Comparison with the exact posterior to quantify the information lost by the compression.

### 8. `08_sbikit.ipynb` (`sbikit` Package)
*   **Topic:** The pipeline of Notebook 7 with [`sbikit`](https://github.com/D-Gebauer/sbikit); also a quickstart for it.
*   **Concepts:**
    *   Compressors, `NLE`, priors, batched posterior sampling with convergence diagnostics.
    *   Calibration metrics with error bars, for all parameters and for subsets.
    *   Changing the prior by importance reweighting, saving models and run records.

### 9. `09_comparison.ipynb` (Side by Side)
*   **Topic:** The same task in all four ways (own code, `sbi`, `ltu-ili`, `sbikit`): code, accuracy and runtime next to each other, and guidance on which tool fits which job.

Two small modules next to the notebooks hold shared code: `notebooks/tutorial_utils.py` (the two toy problems, fixed benchmark simulations, exact posteriors, plotting) and `notebooks/flow_utils.py` (the spline function and the conditional flow from Notebook 3).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/D-Gebauer/SBI_Tutorial.git
    cd SBI_Tutorial
    ```

2.  Create a conda environment (recommended):
    ```bash
    conda create -n sbi_tutorial python=3.13
    conda activate sbi_tutorial
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Notebooks 6, 8 and 9 additionally need `ltu-ili` and `sbikit`, which are installed from GitHub:
    ```bash
    pip install -r requirements-extra.txt
    ```

## Usage

Start the Jupyter server:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and open the notebooks in order.

## Dependencies

*   `numpy`
*   `matplotlib`
*   `torch`
*   `scikit-learn`
*   `sbi`
*   `tarp`
*   `emcee`
*   `chainconsumer`
*   `pandas`
*   `tqdm`
*   `jupyter`
*   `ltu-ili`, `sbikit` (Notebooks 6, 8 and 9 only, see `requirements-extra.txt`)

## References

*   **TARP:** [Lemos et al., 2023](https://arxiv.org/abs/2302.03026)
*   **sbi:** [Tejero-Cantero et al., 2020](https://doi.org/10.21105/joss.02505)
*   **LtU-ILI:** [Ho et al., 2024](https://arxiv.org/abs/2402.05137)
