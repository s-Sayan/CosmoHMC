# File: cosmo_hmc/linear/sampler.py
import cosmohmc.distnd.sampler as sampler
import numpy as np

class linear_post:
    def __init__(self, x_data, y_data, proposal_width=1, anal_grad = False, step_size=0.1, n_steps=10, n_samples=1000):
        """Initializes the class with the data and hyperparameters

        Args:
            x_data (float): data for x
            y_data (float): data for y with same length as x_data
            proposal_width (int, optional): _description_. Defaults to 1.
            anal_grad (bool, optional): use analytical gradients. Defaults to True.
            step_size (float, optional): step size for leapfrog integrator. Defaults to 0.1.
            n_steps (int, optional): number of steps for leapfrog integrator. Defaults to 10.
            n_samples (int, optional): number of samples to draw. Defaults to 1000.
        """
        self.x_data, self.y_data = x_data, y_data
        self.proposal_width = proposal_width
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.anal_grad = anal_grad
        
    def log_likelihood(self, slope, intercept, x, y, sigma=1):
        """ Computes the log likelihood for linear regression
        Args:
            slope (float): slope of the line
            intercept (float): intercept of the line
            x (float): data for x
            y (float): data for y with same length as x
            sigma (float, optional): standard deviation of the noise. Defaults to 1.
        """
        y_pred = slope * x + intercept
        return -0.5 * np.sum(((y - y_pred) / sigma) ** 2)

    def log_posterior(self, slope, intercept, x, y):
        """ Computes the log posterior, with a flat prior """
        return self.log_likelihood(slope, intercept, x, y)
    
    def analytical_gradients(self, slope, intercept, x, y):
        """ Calculate the analytical gradients of the log-likelihood """
        d_log_likelihood_slope = np.sum((y - (slope * x + intercept)) * x)
        d_log_likelihood_intercept = np.sum(y - (slope * x + intercept))
        if self.anal_grad:
            return np.array([d_log_likelihood_slope, d_log_likelihood_intercept])
        else:
            return None    
    
        

    def mcmc_sample(self):
        sampler_mcmc = sampler.mcmcsampler2D(log_prob=lambda x: self.log_posterior(slope=x[0], intercept=x[1], x=self.x_data, y=self.y_data),
                                 proposal_width=self.proposal_width, n_samples=self.n_samples)
        
        mcmc_samples = sampler_mcmc.sample()
        return mcmc_samples, sampler_mcmc.acceptance_rate
    
    def hmc_sample(self):
        if self.anal_grad:
            self.grad_log_prob = lambda x: self.analytical_gradients(slope=x[0], intercept=x[1], x=self.x_data, y=self.y_data)
        else:
            self.grad_log_prob = None
        sampler_hmc = sampler.HMCsampler2D(log_prob=lambda x: self.log_posterior(slope=x[0], intercept=x[1], x=self.x_data, y=self.y_data),
                                           grad_log_prob=self.grad_log_prob,
                                           step_size=self.step_size, n_steps=self.n_steps, n_samples=self.n_samples)
        
        mcmc_samples = sampler_hmc.sample()
        return mcmc_samples, sampler_hmc.acceptance_rate