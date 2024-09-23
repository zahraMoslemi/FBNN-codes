# FBNN-codes

Welcome to **FBNN-codes**! This repository contains tools for simulating datasets, calibrating Bayesian Neural Networks (BNN), and performing emulation and sampling steps for binary classification tasks. The **Fast Bayesian Neural Network (FBNN)** model is a computationally efficient framework designed to perform Bayesian Neural Network inference for high-dimensional data, leveraging a novel **Calibration-Emulation-Sampling (CES)** strategy.

## Quick Start

Run `FBNN/loadResultFBNN.ipynb` to collect posterior samples obtained using the FBNN method.

To reproduce the entire process from the beginning, follow these steps:

### 1. Dataset Simulation using `NnSim.py`

The `NnSim.py` file, located in the `FBNN/bnn/` directory, allows you to simulate datasets using the `make_classification()` function from the sklearn library. The output is saved as `FBNN/bnn/result/nn.pickle`.

### 2. Calibration using `calibration_bnn.ipynb`

The `Baseline-BNN/simulate.ipynb` notebook includes steps to perform calibration by training a BNN. This stage uses **Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)** for parameter sampling. The output is saved as `FBNN/bnn/analysis/calibration_dnn.pickle`. The default is 1 chain with 2000 iterations.

### 3. Emulation and Sampling using `loadResultFBNN.ipynb`

The `loadResultFBNN.ipynb` notebook executes the **FBNN method** to emulate the posterior distribution and collect samples. The emulation step uses a **Deep Neural Network (DNN)** to map parameters to the posterior, drastically improving computational efficiency. Posterior samples are saved in `FBNN/bnn/result`.

## Methodology

The FBNN method adopts a **Calibration-Emulation-Sampling (CES)** strategy, aimed at improving the efficiency of Bayesian inference in neural networks:

1. **Calibration**: Uses **SGHMC** to collect an initial set of posterior samples, serving as training data for the emulator. This step ensures efficient exploration of the parameter space.
   
2. **Emulation**: A DNN is trained to approximate the relationship between the model parameters and posterior likelihood, bypassing the need for expensive posterior evaluations during the sampling stage.

3. **Sampling**: **Preconditioned Crank-Nicolson (pCN)** MCMC algorithm is employed for efficient sampling, leveraging the emulator to reduce computational costs.

## Key Advantages

- **Computational Efficiency**: By emulating the likelihood function using DNNs, the method reduces the time and resources required to perform Bayesian inference on high-dimensional data.
- **Scalability**: FBNN scales Bayesian Neural Networks to high-dimensional problems without compromising accuracy in uncertainty quantification.
- **Accurate Uncertainty Quantification**: The method ensures proper uncertainty estimates, maintaining similar performance to traditional BNNs while significantly improving speed.
