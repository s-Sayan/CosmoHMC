import numpy as np  # Use autograd's wrapped numpy
import cosmohmc.distnd.sampler as sampler

class LinearPost:
    def __init__(self, x_data, y_data, proposal_width=1, anal_grad=False, step_size=0.1, n_steps=10, n_samples=1000, sigma=1):
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
        self.x_data, self.y_data = x_data, y_data
        self.proposal_width = proposal_width
        self.anal_grad = anal_grad
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.dim = 2  # Dimensionality of the parameter space (slope, intercept)
        self.sigma = sigma  # Standard deviation of noise

    def log_likelihood(self, slope, intercept):
        """
        Computes the log-likelihood of the data under a linear model with Gaussian noise.

        Args:
            slope (float): Slope parameter of the linear model.
            intercept (float): Intercept parameter of the linear model.

        Returns:
            float: The log-likelihood of the observed data.
        """
        y_pred = slope * self.x_data + intercept
        return -0.5 * np.sum(((self.y_data - y_pred) / self.sigma) ** 2)

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
        return self.log_likelihood(slope, intercept)

    def analytical_gradients(self, params):
        """
        Calculates the analytical gradients of the log-likelihood with respect to the model parameters.

        Args:
            params (list or np.ndarray): Model parameters [slope, intercept].

        Returns:
            np.ndarray: The gradients [d_log_likelihood/d_slope, d_log_likelihood/d_intercept].
        """
        slope, intercept = params
        y_pred = slope * self.x_data + intercept
        d_log_likelihood_slope = np.sum((self.y_data - y_pred) * self.x_data) / (self.sigma**2)
        d_log_likelihood_intercept = np.sum(self.y_data - y_pred) / (self.sigma**2)
        return np.array([d_log_likelihood_slope, d_log_likelihood_intercept])

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
        grad_log_prob = self.analytical_gradients if self.anal_grad else None
        sampler_hmc = sampler.HMCsamplerN(log_prob=lambda params: self.log_posterior(params),
                                           grad_log_prob=grad_log_prob,
                                           dim=self.dim, step_size=self.step_size, n_steps=self.n_steps, n_samples=self.n_samples)
        
        hmc_samples = sampler_hmc.sample()
        return hmc_samples, sampler_hmc.acceptance_rate


class LinearPost_flex:
    def __init__(self, model_fn, x_data, y_data, proposal_width=1, grad_fn=None, step_size=0.1, n_steps=10, n_samples=1000, sigma=1):
        """
        Initializes the LinearPost class with data, sampling parameters, and model configuration.

        Args:
            x_data (np.ndarray): The independent variable data.
            y_data (np.ndarray): The dependent variable data.
            proposal_width (float, optional): Proposal distribution's standard deviation for MCMC. Defaults to 1.
            grad_fn (callable, optional): Function to compute the gradient of the log probability. If None, gradient is estimated numerically.
            step_size (float, optional): Step size for the leapfrog integrator in HMC. Defaults to 0.1.
            n_steps (int, optional): Number of steps for the leapfrog integrator in HMC. Defaults to 10.
            n_samples (int, optional): Number of samples to draw in both MCMC and HMC. Defaults to 1000.
            sigma (float, optional): Standard deviation of the observation noise. Defaults to 1.
        """
        self.model_fn = model_fn  # Model function
        self.x_data, self.y_data = x_data, y_data
        self.proposal_width = proposal_width
        self.grad_fn = grad_fn  # Gradient of the model function, if available
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.dim = 2  # Dimensionality of the parameter space (slope, intercept)
        self.sigma = sigma  # Standard deviation of noise

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
    
    def analytical_gradients(self, params):
        """Calls the provided gradient function if available; otherwise, returns None."""
        if self.grad_fn is not None:
            return self.grad_fn(self.x_data, self.y_data, params, self.sigma)
        else:
            return None

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
    
        if self.grad_fn is not None:
            # Wrap the gradient function to include all needed arguments
            self.grad_log_prob = lambda params: self.grad_fn(self.x_data, self.y_data, params, self.sigma)
        else:
            self.grad_log_prob = None
        
        # Now, grad_log_prob includes all required arguments and can be called with just `params`
        sampler_hmc = sampler.HMCsamplerN(log_prob=lambda params: self.log_posterior(params),
                                           grad_log_prob=self.grad_log_prob,
                                           dim=self.dim, step_size=self.step_size, n_steps=self.n_steps, n_samples=self.n_samples)
        
        hmc_samples = sampler_hmc.sample()
        return hmc_samples, sampler_hmc.acceptance_rate
    
