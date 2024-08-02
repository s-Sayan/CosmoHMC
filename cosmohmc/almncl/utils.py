# File: cosmohmc/almncl/utils.py

import numpy as np
import healpy as hp

def n2lm_index(n: int) -> tuple[int, int]:
    """
    Converts a one-dimensional index to a two-dimensional (l, m) index.

    Args:
    - n (int): The one-dimensional index of alm.

    Returns:
    - tuple[int, int]: The corresponding (l, m) index.
    """
    lt = (np.sqrt(8.0*n+1)-1.0)/2.0
    l = int(lt)
    m = int(n - l*(l+1)/2)
    return l, m

def lm2n_index(l: int, m: int) -> int:
    """
    Converts a two-dimensional (l, m) index to a one-dimensional index.

    Args:
    - l (int): The spherical harmonic degree.
    - m (int): The order of the spherical harmonic degree.

    Returns:
    - int: The corresponding one-dimensional index.
    """
    return int(l*(l+1)/2 + m)

def convert_alm_to_2d(alm_data, elmax):
    # Calculate the maximum l (elmax) from the length of alm_data
    n_alms = int((elmax+1)*(elmax+2)/2)
    # Ensure alm_data has the correct length
    if len(alm_data) != n_alms:
        raise ValueError(f"alm_data length is incorrect. Expected {n_alms}, got {len(alm_data)}")
    
    # Initialize the 2D array to hold the complex alms and the corresponding 1D real and imaginary parts
    dlm = np.zeros((elmax + 1, elmax + 1), dtype=complex)
    d_almr = np.zeros(n_alms)
    d_almi = np.zeros(n_alms)
    
    # Fill the 2D array from the 1D alm_data
    index = 0
    for m in range(elmax + 1):
        for l in range(m, elmax + 1):
            dlm[l, m] = alm_data[index]
            index += 1
            
    # Extract the real and imaginary parts
    index = 0
    for i in range(elmax + 1):
        for j in range(i + 1):
            d_almr[index] = dlm[i, j].real
            d_almi[index] = dlm[i, j].imag
            index += 1
            
    return d_almr, d_almi

def reconstruct_alm_from_2d(d_almr, d_almi, elmax):
    # Calculate the expected length of d_almr and d_almi based on elmax
    n_alms = (elmax + 1) * (elmax + 2) // 2
    
    # Ensure d_almr and d_almi have the correct length
    if len(d_almr) != n_alms or len(d_almi) != n_alms:
        raise ValueError(f"d_almr and d_almi length is incorrect. Expected {n_alms}, got {len(d_almr)} and {len(d_almi)}")
    
    # Initialize the 2D array for the complex alms
    dlm = np.zeros((elmax + 1, elmax + 1), dtype=complex)
    
    # Fill the 2D array by combining d_almr and d_almi
    index = 0
    for i in range(elmax + 1):
        for j in range(i + 1):
            dlm[i, j] = d_almr[index] + 1j * d_almi[index]
            index += 1
    
    # Flatten the 2D array into 1D alm_data following the healpy convention
    alm_data = []
    for m in range(elmax + 1):
        for l in range(m, elmax + 1):
            alm_data.append(dlm[l, m])
            
    return np.array(alm_data)

def mass_cl(ell: int, Cel: float = 1.0, Nel: float = 1.0) -> float:
    """
    Calculates the mass for Cl parameters.

    Args:
    - ell (int): The spherical harmonic degree.
    - Cel (float, optional): The power spectrum Cl. Defaults to 1.0.
    - Nel (float, optional): The noise Nel. Defaults to 1.0.

    Returns:
    - float: The mass for Cl parameters.
    """
    return 0.5*(2*ell+1)/(Cel+Nel)**2

def n_alms(elmax: int) -> int:
    """
    Calculates the number of alms for a given elmax.

    Args:
    - elmax (int): The maximum spherical harmonic degree.

    Returns:
    - int: The number of alms up to and including elmax.
    """
    return int((elmax+1)*(elmax+2)/2)

def momentum_cl(m_cl: np.ndarray, elmax: int) -> np.ndarray:
    """
    Generates a random momentum array for Cl parameters.

    Args:
    - m_cl (np.ndarray): The mass of Cl parameters.
    - elmax (int): The maximum spherical harmonic degree.

    Returns:
    - np.ndarray: An array of momentum values for Cl.
    """
    return np.sqrt(m_cl) * np.random.normal(0.0, 1.0, elmax+1)

def mass_alm(Cl: np.ndarray, Nl: np.ndarray, n_alms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the mass of the real and imaginary parts of alm.

    Args:
    - Cl (np.ndarray): The power spectrum Cl.
    - Nl (np.ndarray): The noise Nl.
    - n_alms (int): The number of alms.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Arrays representing the mass of the real and imaginary parts of alm.
    """
    m_almr = np.ones(n_alms)
    m_almi = np.ones(n_alms)
    # Exclude alms for l = 0 and l = 1
    for i in range(3, n_alms):
        l, m = n2lm_index(i)
        if m == 0:
            m_almr[i] = 1.0 / Cl[l] + 1.0 / Nl[l]
        else:
            m_almr[i] = m_almi[i] = 2.0 / Cl[l] + 2.0 / Nl[l]
    return m_almr, m_almi

def momentum_alm(m_almr: np.ndarray, m_almi: np.ndarray, n_alms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random momentum arrays for the real and imaginary parts of alm.

    Args:
    - m_almr (np.ndarray): The mass array for the real part of alm.
    - m_almi (np.ndarray): The mass array for the imaginary part of alm.
    - n_alms (int): The number of alms.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Momentum arrays for the real and imaginary parts of alm.
    """
    p_almr = np.sqrt(m_almr) * np.random.normal(0, 1.0, n_alms)
    p_almi = np.sqrt(m_almi) * np.random.normal(0, 1.0, n_alms)
    return p_almr, p_almi

def pdot_alm(q_almr: np.ndarray, q_almi: np.ndarray, d_almr: np.ndarray, d_almi: np.ndarray, q_cl: np.ndarray, Nl: np.ndarray, n_alms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the momentum derivative for the real and imaginary parts of alms.

    Args:
    - q_almr (np.ndarray): The real part of alms.
    - q_almi (np.ndarray): The imaginary part of alms.
    - d_almr (np.ndarray): The real part of data alms.
    - d_almi (np.ndarray): The imaginary part of data alms.
    - q_cl (np.ndarray): The power spectrum derived from alms.
    - Nl (np.ndarray): The noise Nl.
    - n_alms (int): The number of alms.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Momentum derivative arrays for the real and imaginary parts of alms.
    """
    p_almr_d = np.zeros(n_alms)
    p_almi_d = np.zeros(n_alms)
    for k in range(3, n_alms):
        l, m = n2lm_index(k)
        if m == 0:
            p_almr_d[k] = -q_almr[k] / q_cl[l] + (d_almr[k] - q_almr[k]) / Nl[l]
            p_almi_d[k] = 0.0
        else:
            p_almr_d[k] = -2.0*q_almr[k]/q_cl[l] + 2.0*(d_almr[k] - q_almr[k])/Nl[l]
            p_almi_d[k] = -2.0*q_almi[k]/q_cl[l] + 2.0*(d_almi[k] - q_almi[k])/Nl[l]
    return p_almr_d, p_almi_d

def pdot_cl(el: np.ndarray, Cl_hat: np.ndarray, Cl: np.ndarray) -> np.ndarray:
    """
    Calculates the momentum derivative for Cl.

    Args:
    - el (np.ndarray): The spherical harmonic degrees.
    - Cl_hat (np.ndarray): The Cl obtained using alms.
    - Cl (np.ndarray): The power spectrum Cl parameters.

    Returns:
    - np.ndarray: The momentum derivative for Cl.
    """
    elmax = len(Cl) - 1
    p_cl_d = np.zeros(elmax + 1)
    p_cl_d[2:] = 0.5 * (2 * el[2:] + 1.0) * (Cl_hat[2:] / Cl[2:] - 1.0) / Cl[2:]
    return p_cl_d

#define the Hamiltonian
def KE(p_cl: np.ndarray, p_almr: np.ndarray, p_almi: np.ndarray, m_cl: np.ndarray, m_almr: np.ndarray, m_almi: np.ndarray, n_alms: int) -> float:
    """
    Computes the kinetic energy term of the Hamiltonian.

    Args:
    - p_cl (np.ndarray): Momentum for Cl.
    - p_almr (np.ndarray): Momentum for the real part of alm.
    - p_almi (np.ndarray): Momentum for the imaginary part of alm.
    - m_cl (np.ndarray): Mass for Cl.
    - m_almr (np.ndarray): Mass for the real part of alm.
    - m_almi (np.ndarray): Mass for the imaginary part of alm.
    - n_alms (int): The number of alms.

    Returns:
    - float: The kinetic energy.
    """
    term1 = 0.5 * np.sum(p_cl[2:]**2 / m_cl[2:])
    term2 = 0.5 * np.sum(p_almr[3:]**2 / m_almr[3:])
    #term3 = 0.5 * sum(p_almi[k]**2 / m_almi[k] for k in range(3, n_alms) if n2lm_index(k)[1] != 0)
    term3 = 0.0
    for k in np.arange(3, n_alms, 1):
        l,m = n2lm_index(k)
        if m != 0:
            term3 += p_almi[k]**2/m_almi[k]
    term3 = 0.5*term3
    return term1 + term2 + term3

def PE(Cl_hat: np.ndarray, Dl_hat: np.ndarray, Cl: np.ndarray, Nl: np.ndarray) -> float:
    """
    Computes the potential energy term of the Hamiltonian.

    Args:
    - Cl_hat (np.ndarray): The Cl obtained using alms.
    - Dl_hat (np.ndarray): The power spectrum of the difference between data alms and sample alms.
    - Cl (np.ndarray): The power spectrum Cl parameters.
    - Nl (np.ndarray): The noise Nl.

    Returns:
    - float: The potential energy.
    """
    el = np.arange(len(Cl))
    term1 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Dl_hat[2:] / Nl[2:])
    term2 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Cl_hat[2:] / Cl[2:])
    term3 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * np.log(Cl[2:]))
    return term1 + term2 + term3

def Cl_of_almri(almr: np.ndarray, almi: np.ndarray, elmax: int) -> np.ndarray:
    """
    Computes the Cl for the given alms (real and imaginary part separately).

    Args:
    - almr (np.ndarray): The real part of alms.
    - almi (np.ndarray): The imaginary part of alms.
    - elmax (int): The maximum spherical harmonic degree.

    Returns:
    - np.ndarray: The computed Cl.
    """
    Cl_h = np.zeros(elmax + 1)
    for l in range(2, elmax + 1):
        temp = almr[lm2n_index(l, 0)]**2
        for m in range(1, l + 1):
            k = lm2n_index(l, m)
            temp += 2.0 * (almr[k]**2 + almi[k]**2)
        Cl_h[l] = temp / (2.0 * l + 1.0)
    return Cl_h

def PE_pix(Cl_hat: np.ndarray, q_lm: np.ndarray, d_lm: np.ndarray,  Cl: np.ndarray, sigma: np.ndarray, nside: int) -> float:
    """
    Computes the potential energy term of the Hamiltonian.

    Args:
    - Cl_hat (np.ndarray): The Cl obtained using alms.
    - q_lm (np.ndarray): sample q_lm in healpy format.
    - d_lm (np.ndarray): data d_lm in healpy format.
    - Cl (np.ndarray): The power spectrum Cl parameters.
    - sigma (np.ndarray): The noise sigma in pixel space.
    - nside (int): The nside of the healpix map.

    Returns:
    - float: The potential energy.
    """
    el = np.arange(len(Cl))
    #term1 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Dl_hat[2:] / Nl[2:])
    #term1 = 0.5 * np.sum((hp.alm2map(d_lm - q_lm, nside))*(hp.alm2map(d_lm - q_lm, nside))/ sigma / sigma)
    dismod2 = (d_lm - q_lm)* np.conj(d_lm - q_lm)
    term1 = 0.5 * np.sum((hp.alm2map(dismod2, nside))/ sigma / sigma)
    term2 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Cl_hat[2:] / Cl[2:])
    term3 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * np.log(Cl[2:]))
    return term1 + term2 + term3

def PE_pix_v2(Cl_hat: np.ndarray, qlmr: np.ndarray, qlmi: np.ndarray, dlmr: np.ndarray, dlmi: np.ndarray,  Cl: np.ndarray, sigma: np.ndarray, nside: int) -> float:
    """
    Computes the potential energy term of the Hamiltonian.

    Args:
    - Cl_hat (np.ndarray): The Cl obtained using alms.
    - q_lm (np.ndarray): sample q_lm in healpy format.
    - d_lm (np.ndarray): data d_lm in healpy format.
    - Cl (np.ndarray): The power spectrum Cl parameters.
    - sigma (np.ndarray): The noise sigma in pixel space.
    - nside (int): The nside of the healpix map.

    Returns:
    - float: The potential energy.
    """
    el = np.arange(len(Cl))
    #term1 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Dl_hat[2:] / Nl[2:])
    difflm = reconstruct_alm_from_2d(dlmr-qlmr, dlmi-qlmi, elmax = nside)
    
    term1 = 0.5 * np.sum((hp.alm2map(difflm, nside))*(hp.alm2map(difflm, nside))/ sigma / sigma)
    term2 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Cl_hat[2:] / Cl[2:])
    term3 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * np.log(Cl[2:]))
    return term1 + term2 + term3

def PE_pix_v3(Cl_hat: np.ndarray, q_lm: np.ndarray, d_lm: np.ndarray,  Cl: np.ndarray, Nl: np.ndarray) -> float:
    """
    Computes the potential energy term of the Hamiltonian.

    Args:
    - Cl_hat (np.ndarray): The Cl obtained using alms.
    - q_lm (np.ndarray): sample q_lm in healpy format.
    - d_lm (np.ndarray): data d_lm in healpy format.
    - Cl (np.ndarray): The power spectrum Cl parameters.
    - sigma (np.ndarray): The noise sigma in pixel space.
    - nside (int): The nside of the healpix map.

    Returns:
    - float: The potential energy.
    """
    el = np.arange(len(Cl))
    lmax = len(Cl) - 1
    nalms = n_alms(len(Cl)-1)
    #term1 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Dl_hat[2:] / Nl[2:])
    dismod2 = np.real((d_lm - q_lm)* np.conj(d_lm - q_lm))
    dismodr, dismodi = convert_alm_to_2d(dismod2, lmax)
    integrand = np.zeros(lmax+1)
    for l in range(3, lmax+1):
        temp = dismodr[lm2n_index(l, 0)]**2
        for m in range(1, l + 1):
            k = lm2n_index(l, m)
            temp += 2.0 * dismodr[k]
        integrand[l] = temp / Nl[l]
    
    term1 = 0.5 * np.real(np.sum(integrand[3:]))
    term2 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * Cl_hat[2:] / Cl[2:])
    term3 = 0.5 * np.sum((2.0 * el[2:] + 1.0) * np.log(Cl[2:]))
    return term1 + term2 + term3, dismod2
