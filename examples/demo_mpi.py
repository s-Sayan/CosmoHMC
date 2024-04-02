import cosmohmc.distnd.sampler_mpi as sampler
import numpy as np
#from matplotlib import pyplot as plt

def mixture_of_gaussians_log_prob(x):
    mean1 = np.array([2.0, 2.0])
    cov1 = np.array([[1.0, 0.2], [0.2, 1.0]])
    mean2 = np.array([-2.0, -2.0])
    cov2 = np.array([[1.0, -0.2], [-0.2, 1.0]])
    
    log_prob1 = -0.5 * (np.dot((x - mean1).T, np.linalg.inv(cov1)).dot(x - mean1) + np.log(np.linalg.det(cov1)) + 2 * np.log(2 * np.pi))
    log_prob2 = -0.5 * (np.dot((x - mean2).T, np.linalg.inv(cov2)).dot(x - mean2) + np.log(np.linalg.det(cov2)) + 2 * np.log(2 * np.pi))
    
    return np.log(np.exp(log_prob1) + np.exp(log_prob2)) - np.log(2)  # Assuming equal mixture weights

def numerical_grad(f, x, h=1e-5):
    """A general purpose function to compute numerical gradient of f at x."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h_plus = np.array(x)
        x_h_minus = np.array(x)
        x_h_plus[i] += h
        x_h_minus[i] -= h
        grad[i] = (f(x_h_plus) - f(x_h_minus)) / (2 * h)
    return grad

dim = 2  # Dimensionality of the distribution

hmc_sampler_mog = sampler.HMCsamplerN(
    log_prob=lambda x: mixture_of_gaussians_log_prob(x),
    grad_log_prob=lambda x: numerical_grad(mixture_of_gaussians_log_prob, x),
    dim=dim, step_size=0.02, n_steps=50, n_samples=100000
)

mc_sampler_mog = sampler.mcmcsamplerN(
    log_prob=lambda x: mixture_of_gaussians_log_prob(x),
    dim=dim, proposal_width=0.5, n_samples=100000
)

mc_samples_mog = mc_sampler_mog.sample()
hmc_samples_mog = hmc_sampler_mog.sample()

np.savetxt("mog_mc_samples.txt", mc_samples_mog)
np.savetxt("mog_hmc_samples.txt", hmc_samples_mog)