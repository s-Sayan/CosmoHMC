# File: cosmo_hmc/polynom/sampler.py
import cosmohmc.distnd.sampler as sampler
import cosmohmc.almncl.utils as utils
import healpy as hp
import numpy as np
import inspect

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
        
        self.m_almr, self.m_almi = utils.mass_alm(self.Cl, self.Nl, self.elmax)
        
        
        #def HMCsample():
            
        
