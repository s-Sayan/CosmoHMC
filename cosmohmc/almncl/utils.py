import numpy as np
import healpy as hp

def n2lm_index(n):
    """Input: one dimensional index of alm. Output: corresponding (l,m) index"""
    lt = (np.sqrt(8.0*n+1)-1.0)/2.0
    l = int(lt)
    m = int(n-l*(l+1)/2)
    return l,m

def lm2n_index(l,m):
    """Input: (l,m) index of the alm. Output: corresponding one dimensional index."""
    n = int(l*(l+1)/2 + m)
    return n

def mass_cl(ell, Cel = 1.0, Nel = 1.0):
    """Input: el & Cl, for l = 2 to elmax. Output: the mass for Cl parameters."""
    return 0.5*(2*ell+1)/(Cel+Nel)**2

def n_alms(elmax):
    """Input: elmax. Output: number of alms for elmax."""
    return int((elmax+1)*(elmax+2)/2)

def momentum_cl(m_cl, elmax):
    """Input: the mass of Cl, elmax. Output: the (elmax-1) length array of momentum for Cl."""
    return np.sqrt(m_cl)*np.random.normal(0.0, 1.0, elmax+1)

def mass_alm(Cl, Nl, n_alms):
    """Input: Cl, elmax. Output: mass of real and imaginary part of alm"""
    """For the time being, assume data Cl is exact Cl"""
    m_almr = np.ndarray(shape = (n_alms))
    m_almi = np.ndarray(shape = (n_alms))
    m_almr.fill(1.0)
    m_almi.fill(1.0)
    #excluding alms for l = 0, and l = 1
    for i in np.arange(3, n_alms, 1):
        l,m = n2lm_index(i)
        if m == 0:
            #posterior variance of alm, when m = 0
            m_almr[i] = 1.0/Cl[l] + 1.0/Nl[l]
            m_almi[i] = 1.0
        else:
            #posterior variance of alm, when m != 0
            m_almr[i] = 2.0/Cl[l] + 2.0/Nl[l]
            m_almi[i] = 2.0/Cl[l] + 2.0/Nl[l]
    return m_almr, m_almi

def momentum_alm(m_almr, m_almi, n_alms):
    """Input: n_alms, m_almr, m_almi, 
    Output: momenum for real and imaginary part of alm"""
    p_almr = np.sqrt(m_almr)*np.random.normal(0, 1.0, n_alms)
    p_almi = np.sqrt(m_almi)*np.random.normal(0, 1.0, n_alms)
    return p_almr, p_almi

def pdot_alm(q_almr, q_almi, d_almr, d_almi, q_cl, Nl, n_alms):
    """Input:q_almr, q_almi, d_almr, d_almi, q_cl, Nl, n_alms. 
    Output: momentum derivative for real and imaginary part of alms- p_almr_dot & p_almi_dot."""
    p_almr_d = np.ndarray(shape = (n_alms))
    p_almi_d = np.ndarray(shape = (n_alms))
    p_almr_d.fill(0.0)
    p_almi_d.fill(0.0)
    for k in np.arange(3, n_alms, 1):
        l,m = n2lm_index(k)
        if m == 0:
            p_almr_d[k] = -1.0*q_almr[k]/q_cl[l] + (d_almr[k] - q_almr[k])/Nl[l]
            p_almi_d[k] = 0.0
        else:
            p_almr_d[k] = -2.0*q_almr[k]/q_cl[l] + 2.0*(d_almr[k] - q_almr[k])/Nl[l]
            p_almi_d[k] = -2.0*q_almi[k]/q_cl[l] + 2.0*(d_almi[k] - q_almi[k])/Nl[l]
    return p_almr_d, p_almi_d

def pdot_cl(el, Cl_hat, Cl):
    """Input: Cl_hat (Cl obtained using alms) and the Cl parameter. 
    Output: momentum derivative for Cl."""
    elmax = len(Cl)-1
    p_cl_d = np.ndarray(shape = (elmax+1))
    p_cl_d[0] = 0.0
    p_cl_d[1] = 0.0
    p_cl_d[2:] = 0.5*(2*el[2:]+1.0)*(Cl_hat[2:]/Cl[2:] - 1.0)/Cl[2:]
    return p_cl_d

#define the Hamiltonian
def KE(p_cl, p_almr, p_almi, m_cl, m_almr, m_almi, n_alms):
    """Computes the kinetic energy term of the Hamiltonian.
    Input: p_cl, p_almr, p_almi, m_cl, m_almr, m_almi, n_alms.
    Output: Kinetic energy."""
    term1 = 0.5*np.sum(p_cl[2:]**2/m_cl[2:])
    term2 = 0.5*np.sum(p_almr[3:]**2/m_almr[3:])
    term3 = 0.0
    for k in np.arange(3, n_alms, 1):
        l,m = n2lm_index(k)
        if m != 0:
            term3 += p_almi[k]**2/m_almi[k]
    term3 = 0.5*term3
    KE = term1 + term2 + term3
    return KE

#define potential energy
def PE(Cl_hat, Dl_hat, Cl, Nl):
    """Computes the potential energy term of the Hamiltonian.
    Input: Cl_hat (Cl obtained using alms), 
           Dl_hat (Power spectrum of difference between data alms and sample alms)
           Cl, Nl.
    Output: Potential energy."""
    el = np.arange(len(Cl))
    term1 = 0.5*np.sum((2.0*el[2:]+1.0)*Dl_hat[2:]/Nl[2:])
    term2 = 0.5*np.sum((2.0*el[2:]+1.0)*Cl_hat[2:]/Cl[2:])
    term3 = 0.5*np.sum((2.0*el[2:]+1.0)*np.log(Cl[2:]))
    PE = term1 + term2 + term3
    return PE

#compute Cl_hat
def Cl_of_almri(almr, almi, elmax):
    """Computes the Cl for the given alms (real and imaginary part separately)"""
    Cl_h = np.ndarray(shape = (elmax+1))
    Cl_h[0] = 0.0
    Cl_h[1] = 0.0
    for l in np.arange(2, elmax+1, 1):
        temp = 0.0
        k = lm2n_index(l,0)
        temp = almr[k]**2
        for m in np.arange(1, l + 1, 1):
            k = lm2n_index(l,m)
            temp += 2.0*(almr[k]**2 + almi[k]**2)
        Cl_h[l] = temp/(2.0*l + 1.0)
    return Cl_h


