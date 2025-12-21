# Machine Learning & Simulation-Based Inference for Astrophysics

This repository contains a collection of tutorial notebooks designed to introduce **Machine Learning (ML)** and **Simulation-Based Inference (SBI)** to Astrophysics students.

The goal is to build a strong foundational understanding by implementing key concepts from scratch (like Neural Networks and Normalizing Flows) before moving to implementations using libraries.

## Content

The tutorials are organized into four notebooks:

### 1. `01_mlp_from_scratch.ipynb` (NumPy)
*   **Topic:** Building a Multi-Layer Perceptron (MLP) for regression without deep learning frameworks.
*   **Application:** Estimating galaxy distances from redshift (toy model).
*   **Concepts:**
    *   Forward and Backward propagation equations.
    *   Activation functions (ReLU, LeakyReLU, Sigmoid).
    *   Loss functions (MSE) and Regularization (L1/L2).
    *   Optimizers implementation (SGD, Momentum, Adam).

### 2. `02_torch_regression.ipynb` (PyTorch)
*   **Topic:** Introduction to PyTorch for building modular neural networks.
*   **Application:** Redshift-Distance regression (revisited).
*   **Concepts:**
    *   PyTorch Tensors and Autograd.
    *   `nn.Module` class structure.
    *   `DataLoader` and efficient batching.
    *   Training loops with `torch.optim`.

### 3. `03_normalizing_flows.ipynb` (Flows from Scratch)
*   **Topic:** Density estimation using Normalizing Flows.
*   **Application:** Modeling the "Two Moons" dataset.
*   **Concepts:**
    *   Change of variables formula and Jacobian determinant.
    *   **RealNVP** (Real Non-Volume Preserving) Coupling Layers.
    *   **MAF** (Masked Autoregressive Flow) using MADE (Masked Autoencoder).
    *   **NSF** (Neural Spline Flow) with simplified Rational Quadratic Splines.
    *   Implemented from scratch using `torch.nn` primitives.

### 4. `04_sbi_inference.ipynb` (SBI Package)
*   **Topic:** Likelihood-free inference for simulations.
*   **Application:** Parameter inference on a Gaussian toy model.
*   **Concepts:**
    *   **NPE** (Neural Posterior Estimation).
    *   **NLE** (Neural Likelihood Estimation).
    *   Using the [`sbi`](https://sbi-dev.github.io/sbi/) package.
    *   Posterior Validation and Coverage Tests using **TARP** ([`tarp`](https://github.com/Ciela-Institute/tarp)).

### 5. `05_sbi_advanced.ipynb` (SBI Package)
*   **Topic:** Advanced SBI methods for complex simulations.
*   **Application:** Parameter inference on a Non-Gaussian toy model.
*   **Concepts:**
    *   Custom Density Estimators.
    *   Custom Training Loops.

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
*   `jupyter`

## References

*   **TARP:** [Lemos et al., 2023](https://arxiv.org/abs/2302.03026)
