# File: cosmohmc/dist1d/sampler.py
import numpy as np

class mcmcsampler:
    def __init__(self, log_prob, proposal_width=1, n_samples=1000):
        """
        Initialize the MCMC sampler for a generic distribution.

        Args:
            log_prob (callable): Function to compute the log probability of the target distribution.
            proposal_width (float, optional): The standard deviation of the proposal distribution. Default is 1.
            n_samples (int, optional): The number of samples to draw. Default is 1000.
        """
        self.log_prob = log_prob
        self.proposal_width = proposal_width
        self.n_samples = n_samples
        self.acceptance_rate = 0  # Initialize acceptance rate
        self.current_state = np.random.randn()  # Initial state

    def propose_new_state(self):
        """Propose a new state based on the current state and proposal distribution."""
        return np.random.normal(self.current_state, self.proposal_width)

    def sample(self):
        """Generate samples using the Metropolis-Hastings algorithm."""
        samples = []
        accepted = 0  # Count of accepted proposals

        for _ in range(self.n_samples):
            proposal = self.propose_new_state()
            
            # Calculate acceptance probability
            current_log_prob = self.log_prob(self.current_state)
            proposal_log_prob = self.log_prob(proposal)
            accept_prob = np.exp(proposal_log_prob - current_log_prob)  # Ratio of probabilities
            
            # Accept or reject the proposal
            if np.random.rand() < accept_prob:
                self.current_state = proposal
                accepted += 1
            
            samples.append(self.current_state)

        # Calculate acceptance rate
        self.acceptance_rate = accepted / self.n_samples
        return samples
    

class HMCsampler:
    def __init__(self, log_prob, grad_log_prob=None, step_size=0.1, n_steps=10, n_samples=1000):
        """
        Initialize the HMC sampler for a generic distribution.

        Args:
            log_prob (callable): Function to compute the log probability of the target distribution.
            grad_log_prob (callable, optional): Function to compute the gradient of the log probability. If None, the gradient will be estimated numerically.
            step_size (float, optional): Step size for the leapfrog integrator. Default is 0.1.
            n_steps (int, optional): Number of steps for the leapfrog integrator. Default is 10.
            n_samples (int, optional): Number of samples to draw. Default is 1000.
        """
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.acceptance_rate = 0
        self.current_state = np.random.randn()

    def numerical_grad_log_prob(self, x, h=1e-5):
        """Estimate the gradient of the log probability function numerically."""
        grad = (self.log_prob(x + h) - self.log_prob(x - h)) / (2 * h)
        return grad

    def leapfrog(self, x, p):
        """Perform leapfrog steps to simulate Hamiltonian dynamics."""
        x_new, p_new = np.copy(x), np.copy(p)

        if self.grad_log_prob is None:
            grad_func = self.numerical_grad_log_prob
        else:
            grad_func = self.grad_log_prob

        # Half step for momentum at the beginning
        p_new += 0.5 * self.step_size * grad_func(x_new)

        # Full steps for position and momentum
        for _ in range(self.n_steps - 1):
            x_new += self.step_size * p_new  # Full step for position
            p_new += self.step_size * grad_func(x_new)  # Full step for momentum

        # Half step for momentum at the end
        x_new += self.step_size * p_new
        p_new += 0.5 * self.step_size * grad_func(x_new)

        return x_new, p_new

    def sample(self):
        """Generate samples using the Hamiltonian Monte Carlo algorithm."""
        samples = []
        accepted = 0

        for _ in range(self.n_samples):
            current_momentum = np.random.randn()
            proposed_state, proposed_momentum = self.leapfrog(self.current_state, current_momentum)

            # Hamiltonian calculation for current and proposed state
            current_H = -self.log_prob(self.current_state) + 0.5 * np.sum(current_momentum**2)
            proposed_H = -self.log_prob(proposed_state) + 0.5 * np.sum(proposed_momentum**2)

            # Acceptance probability
            accept_prob = np.exp(current_H - proposed_H)

            # Accept or reject the proposal
            if np.random.rand() < accept_prob:
                self.current_state = proposed_state
                accepted += 1

            samples.append(self.current_state)

        self.acceptance_rate = accepted / self.n_samples
        return samples
    
class mcmcsampler2D:
    def __init__(self, log_prob, proposal_width=1, n_samples=1000):
        """
        Initialize the MCMC sampler for a generic 2D distribution.

        Args:
            log_prob (callable): Function to compute the log probability of the target distribution.
            proposal_width (float, optional): Standard deviation of the proposal distribution. Default is 1.
            n_samples (int, optional): Number of samples to draw. Default is 1000.
        """
        self.log_prob = log_prob
        self.proposal_width = proposal_width
        self.n_samples = n_samples
        self.acceptance_rate = 0
        self.current_state = np.random.randn(2)  # Initial state in 2D

    def propose_new_state(self):
        """Propose a new state based on the current state and proposal distribution."""
        return self.current_state + np.random.normal(scale=self.proposal_width, size=2)

    def sample(self):
        """Generate samples using the Metropolis-Hastings algorithm."""
        samples = []
        accepted = 0

        for _ in range(self.n_samples):
            proposal = self.propose_new_state()
            
            # Calculate acceptance probability
            current_log_prob = self.log_prob(self.current_state)
            proposal_log_prob = self.log_prob(proposal)
            accept_prob = np.exp(proposal_log_prob - current_log_prob)
            
            # Accept or reject the proposal
            if np.random.rand() < accept_prob:
                self.current_state = proposal
                accepted += 1
            
            samples.append(self.current_state.copy())

        self.acceptance_rate = accepted / self.n_samples
        return np.array(samples)
    
class HMCsampler2D:
    def __init__(self, log_prob, grad_log_prob=None, step_size=0.1, n_steps=10, n_samples=1000):
        """
        Initialize the HMC sampler for a generic 2D distribution.

        Args:
            log_prob (callable): Function to compute the log probability.
            grad_log_prob (callable, optional): Function to compute the gradient of the log probability. If None, gradient is estimated numerically.
            step_size (float, optional): Step size for the leapfrog integrator. Default is 0.1.
            n_steps (int, optional): Number of steps for the leapfrog integrator. Default is 10.
            n_samples (int, optional): Number of samples to draw. Default is 1000.
        """
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob if grad_log_prob is not None else self.numerical_grad_log_prob
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.acceptance_rate = 0
        self.current_state = np.random.randn(2)  # Initial state in 2D

    def numerical_grad_log_prob(self, x, h=1e-5):
        """Numerically estimate the gradient of the log probability function."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_h = np.array(x)
            x_h[i] += h
            f_x_h = self.log_prob(x_h)
            x_h[i] -= 2*h
            f_x_h_minus = self.log_prob(x_h)
            grad[i] = (f_x_h - f_x_h_minus) / (2 * h)
        return grad

    def leapfrog(self, x, p):
        """Perform leapfrog steps to simulate Hamiltonian dynamics."""
        x_new, p_new = np.copy(x), np.copy(p)

        # Half step for momentum at the beginning
        p_new += 0.5 * self.step_size * self.grad_log_prob(x_new)

        # Full steps for position and momentum
        for _ in range(self.n_steps - 1):
            x_new += self.step_size * p_new
            p_new += self.step_size * self.grad_log_prob(x_new)

        # Half step for momentum at the end
        x_new += self.step_size * p_new
        p_new += 0.5 * self.step_size * self.grad_log_prob(x_new)

        return x_new, p_new

    def sample(self):
        """Generate samples using the Hamiltonian Monte Carlo algorithm."""
        samples = []
        accepted = 0

        for _ in range(self.n_samples):
            current_momentum = np.random.randn(2)
            proposed_state, proposed_momentum = self.leapfrog(self.current_state, current_momentum)

            # Hamiltonian calculation for current and proposed state
            current_H = -self.log_prob(self.current_state) + 0.5 * np.sum(current_momentum**2)
            proposed_H = -self.log_prob(proposed_state) + 0.5 * np.sum(proposed_momentum**2)

            # Acceptance probability
            accept_prob = np.exp(current_H - proposed_H)

            # Accept or reject the proposal
            if np.random.rand() < accept_prob:
                self.current_state = proposed_state
                accepted += 1

            samples.append(self.current_state.copy())

        self.acceptance_rate = accepted / self.n_samples
        return np.array(samples)
