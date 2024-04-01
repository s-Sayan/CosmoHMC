import autograd.numpy as np  # Use autograd's wrapped numpy
from autograd import grad
import cosmohmc.distnd.sampler as sampler
    
class LinearPost_autograd:
    def __init__(self, model_fn, x_data, y_data, proposal_width=1, step_size=0.1, n_steps=10, n_samples=1000, sigma=1):
        """
        Initializes the LinearPost class with data, sampling parameters, and model configuration.

        Args:
            x_data (np.ndarray): The independent variable data.
            y_data (np.ndarray): The dependent variable data.
            proposal_width (float, optional): Proposal distribution's standard deviation for MCMC. Defaults to 1.
            anal_grad (bool, optional): Flag to use analytical gradients if True; otherwise, None is used. Defaults to False.
            step_size (float, optional): Step size for the leapfrog integrator in HMC. Defaults to 0.1.
            n_steps (int, optional): Number of steps for the leapfrog integrator in HMC. Defaults to 10.
            n_samples (int, optional): Number of samples to draw in both MCMC and HMC. Defaults to 1000.
            sigma (float, optional): Standard deviation of the observation noise. Defaults to 1.
        """
        self.model_fn = model_fn  # Model function
        self.x_data, self.y_data = x_data, y_data
        self.proposal_width = proposal_width
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.dim = 2  # Dimensionality of the parameter space (slope, intercept)
        self.sigma = sigma  # Standard deviation of noise
        self.grad_log_prob = grad(self.log_likelihood_wrapped) # autograd gradient of the log likelihood

    def log_likelihood(self, params):
        """
        Computes the log-likelihood of the data under a linear model with Gaussian noise.

        Args:
            slope (float): Slope parameter of the linear model.
            intercept (float): Intercept parameter of the linear model.

        Returns:
            float: The log-likelihood of the observed data.
        """
        y_pred = self.model_fn(self.x_data, *params)
        return -0.5 * np.sum(((self.y_data - y_pred) / self.sigma) ** 2)
    
    def log_likelihood_wrapped(self, params):
        # Wrap the log likelihood to be compatible with autograd
        # autograd requires the function to take a flat array as input
        return self.log_likelihood(params)

    def log_posterior(self, params):
        """
        Computes the log posterior of the model, assuming uniform priors for simplicity.

        Args:
            params (list or np.ndarray): Model parameters [slope, intercept].

        Returns:
            float: The log posterior probability of the parameters.
        """
        slope, intercept = params
        # Here, we're using a flat prior, so the log-posterior is proportional to the log-likelihood
        return self.log_likelihood(params)
    

    def mcmc_sample(self):
        """
        Performs MCMC sampling of the parameter space.

        Returns:
            np.ndarray: Array of sampled parameters.
            float: Acceptance rate of the MCMC sampler.
        """
        sampler_mcmc = sampler.mcmcsamplerN(log_prob=lambda params: self.log_posterior(params),
                                 dim=self.dim, proposal_width=self.proposal_width, n_samples=self.n_samples)
        
        mcmc_samples = sampler_mcmc.sample()
        return mcmc_samples, sampler_mcmc.acceptance_rate
    
    def hmc_sample(self):
        """
        Performs HMC sampling of the parameter space.

        Returns:
            np.ndarray: Array of sampled parameters.
            float: Acceptance rate of the HMC sampler.
        """
        
        # Now, grad_log_prob includes all required arguments and can be called with just `params`
        sampler_hmc = sampler.HMCsamplerN(log_prob=lambda params: self.log_posterior(params),
                                           grad_log_prob=self.grad_log_prob,
                                           dim=self.dim, step_size=self.step_size, n_steps=self.n_steps, n_samples=self.n_samples)
        
        hmc_samples = sampler_hmc.sample()
        return hmc_samples, sampler_hmc.acceptance_rate