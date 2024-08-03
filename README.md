<img src="https://github.com/s-Sayan/CosmoHMC/blob/main/figure/cosmo_HMC.png" width="1000" alt="CosmoHMC Logo">
CosmoHMC is a Python package providing a Hamiltonian Monte Carlo (HMC) sampler tailored specifically for cosmological data analysis, especially for the Cosmic Microwave Background (CMB). It is designed to efficiently sample from joint posterior distributions of the spherical harmonic coefficients of the signal CMB sky, the power-spectrum (and off-diagonal terms in the covariance matrix due to motion of our observation frame).

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
## Support and Acknowledgement
The work has been supported by Raman Reseach Institute (RRI), Indian Institute of Science Education and Research (IISER), Pune, University of Geneva and Swiss Government Excellence Scholarship (ESKAS No. 2022.0312)\
<img src="https://github.com/s-Sayan/CosmoHMC/blob/main/figure/rri-phd-admissions-2020.webp" width="85" alt="CosmoHMC Logo">
<img src="https://github.com/s-Sayan/CosmoHMC/blob/main/figure/IISER Pune.png" width="150" alt="CosmoHMC Logo">
<img src="https://github.com/s-Sayan/CosmoHMC/blob/main/figure/unige.png" width="150" alt="CosmoHMC Logo">
