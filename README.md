# FBNN-codes

Welcome to FBNN-codes! This repository contains tools for simulating datasets, calibrating Bayesian Neural Networks (BNN), and performing emulation and sampling steps for Binary Classification task. Below are the instructions to get started with each component:

## Quick Start

Simply run `FBNN/loadResultFBNN.ipynb` to collect posterior samples obtained by FBNN method. 

To reproduce the whole process from the beginning, please run the following 3 steps:

## 1. Dataset Simulation using NnSim.py

The `NnSim.py` file, located in the `FBNN/bnn/` directory, allows you to simulate datasets using make_classification() function available in sklearn.datasets library in python. The output is `FBNN/bnn/result/nn.pickle`.

## 2. Calibration using calibration_bnn.ipynb

The `Baseline-BNN/simulate.ipynb` notebook contains the steps to get the calibration data by training BNN. The output is `FBNN/bnn/analysis/calibration_dnn.pickle`. The default is 1 chain with 2000 iterations.

## 3. Emulation and Sampling using loadResultFBNN.ipynb

The `loadResultFBNN.ipynb` notebook allows us to perform FBNN method on saved samples in Calibration stage and collect posterior samples at the end. The emulator will be located in `FBNN/bnn/train_NN`, and the posterior samples are located in `FBNN/bnn/result`.