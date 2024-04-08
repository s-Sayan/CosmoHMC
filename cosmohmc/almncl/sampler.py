# File: cosmo_hmc/polynom/sampler.py
import cosmohmc.distnd.sampler as sampler
import cosmohmc.almncl.utils as utils
import healpy as hp
import numpy as np
from tqdm import tqdm

#import inspect

class almnclsampler():
    def __init__(self, data, elmax, nside, Cl, Nl, step_size = 0.1, n_steps = 10, n_samples = 1000):
        """
        Initializes the almnclsampler class with data, sampling parameters.

        Args:
            data (array): The CMB data to be analyzed.
            elmmax (int): The maximum l value for the spherical harmonic transform.
            nside (int): The nside parameter for healpy.
            Cl (array): The theoretical C_l values for the model.
            Nl (array): The noise power spectrum.
            step_size (float): The step size for the leapfrog integrator.
            nsteps (int): The number of steps for the leapfrog integrator.
            nsamples (int): The number of samples to generate.
        """
        self.data = data
        self.elmax = elmax
        self.nside = nside
        self.Cl = Cl
        self.Nl = Nl
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_samples = n_samples

        self.n_alms = utils.n_alms(self.elmax)
        self.alm_data = hp.map2alm(self.data, lmax = self.elmax)
        self.data_cl = hp.anafast(self.data, lmax = self.elmax)
        self.d_almr = np.ndarray(shape = (self.n_alms))
        self.d_almi = np.ndarray(shape = (self.n_alms))
        self.dlm = np.ndarray(shape=(self.elmax+1, self.elmax+1), dtype = complex)
        
        #first read the 1D indexed alms of healpy, arranged according to healpy convention
        index = 0
        for m in range(self.elmax+1):
            for l in range(m, self.elmax+1):
                self.dlm[l, m] = self.alm_data[index]
                index += 1
                
        #now convert the 1D indexed alms to 2D indexed alms
        index = 0
        for i in range(self.elmax+1):
            for j in range(i+1):
                self.d_almr[index] = self.dlm[i, j].real
                self.d_almi[index] = self.dlm[i, j].imag
                index += 1
         
        self.el = np.arange(self.elmax+1)        
        self.m_cl = np.ndarray(shape = (self.elmax+1))
        self.m_cl.fill(1.0)
        self.m_cl[2:] = utils.mass_cl(self.el[2:], self.Cl[2:], self.Nl[2:])
        
        self.m_almr, self.m_almi = utils.mass_alm(self.Cl, self.Nl, self.n_alms)
        
        
        
    def leapfrog(self, p_cl, p_almr, p_almi, q_cl, q_almr, q_almi):
        """
        The leapfrog integrator for the Hamiltonian Monte Carlo sampler.

        Args:
            p_cl (array): The momentum for the C_l values.
            p_almr (array): The momentum for the real part of the alms.
            p_almi (array): The momentum for the imaginary part of the alms.
            q_cl (array): The position for the C_l values.
            q_almr (array): The position for the real part of the alms.
            q_almi (array): The position for the imaginary part of the alms.

        Returns:
            p_cl (array): The updated momentum for the C_l values.
            p_almr (array): The updated momentum for the real part of the alms.
            p_almi (array): The updated momentum for the imaginary part of the alms.
            q_cl (array): The updated position for the C_l values.
            q_almr (array): The updated position for the real part of the alms.
            q_almi (array): The updated position for the imaginary part of the alms.
        """
        
        el = np.arange(self.elmax+1)
        for _ in range(self.n_steps):
            # half momnetum step
            Cl_hat = utils.Cl_of_almri(q_almr, q_almi, self.elmax)
            p_cl_dot = utils.pdot_cl(el, Cl_hat, q_cl)
            p_almr_dot, p_almi_dot = utils.pdot_alm(q_almr, q_almi, self.d_almr, self.d_almi, q_cl, self.Nl, self.n_alms)
            p_cl = p_cl + 0.5*self.step_size*p_cl_dot
            p_almr = p_almr + 0.5*self.step_size*p_almr_dot
            p_almi = p_almi + 0.5*self.step_size*p_almi_dot
            
            # full position step
            q_cl = q_cl + self.step_size * p_cl/self.m_cl
            q_almr = q_almr + self.step_size*p_almr/self.m_almr
            q_almi = q_almi + self.step_size*p_almi/self.m_almi
            
            # half momentum step
            Cl_hat = utils.Cl_of_almri(q_almr, q_almi, self.elmax)
            p_cl_dot = utils.pdot_cl(el, Cl_hat, q_cl)
            p_almr_dot, p_almi_dot = utils.pdot_alm(q_almr, q_almi, self.d_almr, self.d_almi, q_cl, self.Nl, self.n_alms)
            p_cl = p_cl + 0.5*self.step_size*p_cl_dot
            p_almr = p_almr + 0.5*self.step_size*p_almr_dot
            p_almi = p_almi + 0.5*self.step_size*p_almi_dot
            
        return p_cl, p_almr, p_almi, q_cl, q_almr, q_almi
    
    def leapfrog_v2(self, p_cl, p_almr, p_almi, q_cl, q_almr, q_almi):
        """
        The leapfrog integrator for the Hamiltonian Monte Carlo sampler.

        Args:
            p_cl (array): The momentum for the C_l values.
            p_almr (array): The momentum for the real part of the alms.
            p_almi (array): The momentum for the imaginary part of the alms.
            q_cl (array): The position for the C_l values.
            q_almr (array): The position for the real part of the alms.
            q_almi (array): The position for the imaginary part of the alms.

        Returns:
            p_cl (array): The updated momentum for the C_l values.
            p_almr (array): The updated momentum for the real part of the alms.
            p_almi (array): The updated momentum for the imaginary part of the alms.
            q_cl (array): The updated position for the C_l values.
            q_almr (array): The updated position for the real part of the alms.
            q_almi (array): The updated position for the imaginary part of the alms.
        """
        
        el = np.arange(self.elmax+1)
        for _ in range(self.n_steps):
            
            # half position step
            q_cl_dot = p_cl/self.m_cl
            q_almr_dot = p_almr/self.m_almr
            q_almi_dot = p_almi/self.m_almi
            q_cl = q_cl + 0.5 * self.step_size * q_cl_dot
            q_almr = q_almr + 0.5 * self.step_size * q_almr_dot
            q_almi = q_almi + 0.5 * self.step_size * q_almi_dot
            
            # full momentum step
            Cl_hat = utils.Cl_of_almri(q_almr, q_almi, self.elmax)
            p_cl_dot = utils.pdot_cl(el, Cl_hat, q_cl)
            p_almr_dot, p_almi_dot = utils.pdot_alm(q_almr, q_almi, self.d_almr, self.d_almi, q_cl, self.Nl, self.n_alms)
            p_cl = p_cl + self.step_size*p_cl_dot
            p_almr = p_almr + self.step_size*p_almr_dot
            p_almi = p_almi + self.step_size*p_almi_dot
            
            # half position step
            q_cl_dot = p_cl/self.m_cl
            q_almr_dot = p_almr/self.m_almr
            q_almi_dot = p_almi/self.m_almi
            q_cl = q_cl + 0.5 * self.step_size * q_cl_dot
            q_almr = q_almr + 0.5 * self.step_size * q_almr_dot
            q_almi = q_almi + 0.5 * self.step_size * q_almi_dot
            
        return p_cl, p_almr, p_almi, q_cl, q_almr, q_almi
            
            
    def HMCsample(self):
        """Generate samples using the Hamiltonian Monte Carlo algorithm."""
        
        #Initialise the loop
        q_cl = np.copy(self.data_cl)
        q_almr = np.copy(self.d_almr)
        q_almi = np.copy(self.d_almi)

        #just as a placeholder value
        q_cl_star = np.copy(q_cl)
        q_almr_star = np.copy(q_almr)
        q_almi_star = np.copy(q_almi)
        
        Cl_hat = np.copy(q_cl)
        
        sample_q_almr = []
        sample_q_almi = []
        sample_q_cl = []
        dH_array = []
        KE_term0_array = []
        PE_term0_array = []
        KE_term_star_array = []
        PE_term_star_array = []
        
        accepted = 0
        
        for _ in tqdm(range(self.n_samples), desc="Sampling (HMC)"):
            
            p_almr, p_almi = utils.momentum_alm(self.m_almr, self.m_almi, self.n_alms)
            p_cl = utils.momentum_cl(self.m_cl, self.elmax)
            
            KE_term0 = utils.KE(p_cl, p_almr, p_almi, self.m_cl, self.m_almr, self.m_almi, self.n_alms)
            
            Dl_hat = utils.Cl_of_almri(self.d_almr - q_almr, self.d_almi-q_almi, self.elmax)
            Cl_hat = utils.Cl_of_almri(q_almr, q_almi, self.elmax)
            PE_term0 = utils.PE(Cl_hat, Dl_hat, q_cl, self.Nl)
            
            H0 = KE_term0 + PE_term0
            
            #Leapfrog integration
            #p_cl_star, p_almr_star, p_almi_star, q_cl_star, q_almr_star, q_almi_star = self.leapfrog(p_cl, p_almr, p_almi, q_cl, q_almr, q_almi)            
            p_cl_star, p_almr_star, p_almi_star, q_cl_star, q_almr_star, q_almi_star = self.leapfrog_v2(p_cl, p_almr, p_almi, q_cl, q_almr, q_almi)
            
            KE_term_star = utils.KE(p_cl_star, p_almr_star, p_almi_star, self.m_cl, self.m_almr, self.m_almi, self.n_alms)
            Dl_hat = utils.Cl_of_almri(self.d_almr - q_almr_star, self.d_almi - q_almi_star, self.elmax)
            Cl_hat = utils.Cl_of_almri(q_almr_star, q_almi_star, self.elmax)
            PE_term_star = utils.PE(Cl_hat, Dl_hat, q_cl_star, self.Nl)
            
            H_star = KE_term_star + PE_term_star
            dH = H_star - H0
            accpt_prob = np.exp(-dH)
            
            # Accpect or reject
            if np.random.rand() < accpt_prob:
                q_cl = q_cl_star
                q_almr = q_almr_star
                q_almi = q_almi_star
                accepted += 1
                
            sample_q_cl.append(q_cl)
            sample_q_almr.append(q_almr)
            sample_q_almi.append(q_almi)
            dH_array.append(dH)
            KE_term0_array.append(KE_term0)
            PE_term0_array.append(PE_term0)
            KE_term_star_array.append(KE_term_star)
            PE_term_star_array.append(PE_term_star)
            
        self.acceptance_rate = accepted / self.n_samples
        self.dH_array = dH_array
        self.KE_term0_array = KE_term0_array
        self.PE_term0_array = PE_term0_array
        self.KE_term_star_array = KE_term_star_array
        self.PE_term_star_array = PE_term_star_array
        return np.array(sample_q_cl), np.array(sample_q_almr), np.array(sample_q_almi)
                
        
