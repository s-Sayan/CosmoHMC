# File: cosmohmc/dist1d/sampler.py
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
    
class mcmcsamplerN:
    def __init__(self, log_prob, dim=2, proposal_width=1, n_samples=1000, base_seed=42):
        self.log_prob = log_prob
        self.dim = dim
        self.proposal_width = proposal_width
        self.n_samples = n_samples
        self.acceptance_rate = 0
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Seed the random number generator differently in each MPI process
        np.random.seed(base_seed + self.rank)

        # Initialize the current state with a random start for each process
        self.current_state = np.random.randn(dim)

    def propose_new_state(self):
        return self.current_state + np.random.normal(scale=self.proposal_width, size=self.dim)

    def sample(self):
        local_samples = []
        local_accepted = 0

        # Adjust the number of samples per process
        samples_per_process = self.n_samples // self.size
        extra = self.n_samples % self.size
        if self.rank < extra:
            samples_per_process += 1

        # Sampling loop
        iterator = range(samples_per_process)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Sampling (MCMC)")

        for _ in iterator:
            proposal = self.propose_new_state()
            current_log_prob = self.log_prob(self.current_state)
            proposal_log_prob = self.log_prob(proposal)
            accept_prob = np.exp(proposal_log_prob - current_log_prob)

            if np.random.rand() < accept_prob:
                self.current_state = proposal
                local_accepted += 1

            local_samples.append(self.current_state.copy())

        # Gather all samples at rank 0
        all_samples = self.comm.gather(local_samples, root=0)
        total_accepted = self.comm.reduce(local_accepted, op=MPI.SUM, root=0)

        if self.rank == 0:
            all_samples = np.concatenate(all_samples)
            self.acceptance_rate = total_accepted / self.n_samples
            return all_samples


class HMCsamplerN:
    def __init__(self, log_prob, grad_log_prob=None, dim=2, step_size=0.1, n_steps=10, n_samples=1000, base_seed=42):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob if grad_log_prob is not None else self.numerical_grad_log_prob
        self.dim = dim
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.acceptance_rate = 0
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # Seed the random number generator
        np.random.seed(base_seed + self.rank)
        # Initialize the current state with a different random seed for each process
        self.current_state = np.random.randn(dim)

    def numerical_grad_log_prob(self, x, h=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_h_plus = np.array(x)
            x_h_minus = np.array(x)
            x_h_plus[i] += h
            x_h_minus[i] -= h
            grad[i] = (self.log_prob(x_h_plus) - self.log_prob(x_h_minus)) / (2 * h)
        return grad

    def leapfrog(self, x, p):
        x_new, p_new = np.copy(x), np.copy(p)
        p_new += 0.5 * self.step_size * self.grad_log_prob(x_new)

        for _ in range(self.n_steps - 1):
            x_new += self.step_size * p_new
            p_new += self.step_size * self.grad_log_prob(x_new)

        x_new += self.step_size * p_new
        p_new += 0.5 * self.step_size * self.grad_log_prob(x_new)

        return x_new, p_new

    def sample(self):
        local_samples = []
        local_accepted = 0

        # Determine samples per process
        samples_per_process = self.n_samples // self.size
        extra = self.n_samples % self.size
        if self.rank < extra:
            samples_per_process += 1

        # Progress bar only on rank 0
        iterator = range(samples_per_process)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Sampling (HMC)")

        for _ in iterator:
            current_momentum = np.random.randn(self.dim)
            proposed_state, proposed_momentum = self.leapfrog(self.current_state, current_momentum)
            current_H = -self.log_prob(self.current_state) + 0.5 * np.sum(current_momentum**2)
            proposed_H = -self.log_prob(proposed_state) + 0.5 * np.sum(proposed_momentum**2)
            accept_prob = np.exp(current_H - proposed_H)

            if np.random.rand() < accept_prob:
                self.current_state = proposed_state
                local_accepted += 1

            local_samples.append(self.current_state.copy())

        # Gather all samples at rank 0
        all_samples = self.comm.gather(local_samples, root=0)
        total_accepted = self.comm.reduce(local_accepted, op=MPI.SUM, root=0)

        if self.rank == 0:
            all_samples = np.concatenate(all_samples)
            self.acceptance_rate = total_accepted / self.n_samples
            return all_samples

