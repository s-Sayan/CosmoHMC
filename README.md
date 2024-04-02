# CosmoHMC

![CosmoHMC Logo](https://github.com/s-Sayan/CosmoHMC/blob/main/figure/cosmohmc_logo.png)

CosmoHMC is a Python package providing a Hamiltonian Monte Carlo (HMC) sampler tailored specifically for cosmological data analysis. It is designed to efficiently sample from posterior distributions of cosmological parameters.

## Installation

To install CosmoHMC, follow these simple steps:

### Prerequisites

Ensure you have Python 3.6 or higher installed on your system. You can check your Python version by running:

```bash
python --version
```

### Installation

To install CosmoHMC from the source, first, clone the repository:

```bash
git clone https://github.com/s-Sayan/CosmoHMC.git
cd CosmoHMC
pip install -e .
```
## Usage

The basic implementation is very simple, first you initiate the class and then sample from the posterior distribution:

```python
import cosmohmc.distnd.sampler as sampler
import numpy as np

def gaussian_log_prob(x, mean=0, variance=1):
    """Compute log probability of Gaussian with given mean and variance at x."""
    return -0.5 * np.log(2 * np.pi * variance) - (x - mean)**2 / (2 * variance)

def gaussian_grad_log_prob(x, mean=0, variance=1):
    return -(x - mean) / variance

hmcsmplr = sampler.HMCsampler(log_prob=lambda x: gaussian_log_prob(x, 0, 1),grad_log_prob=lambda x: gaussian_grad_log_prob(x),
                     step_size=0.1, n_steps=10, n_samples=10000)

hmcsamples = hmcsmplr.sample()
```
Other than this, CosmoHMC also has the basic Metropolis-Hastings sampler implemented. For basic posterior distribution sampling, please refer to `demo_linear.ipynb` and `demo_polynom.ipynb` in the examples folder.

### MPI Parallelization
 
CosmoHMC also supports MPI parallelization. To run the sampler in parallel, you need to install mpi4py. You can install it using pip:

```bash
pip install mpi4py
```
Then you can look at the ```demo_mpi.py``` file in the examples folder to see how to run the sampler in parallel. To run the demo_mpi.py file, you can use the following command:

```bash
mpiexec -np 4 python demo_mpi.py
```
