import numpy as np
from numba import jit
from Sersic_profile_functions import *
from scipy.special import gamma, erf, spence


#nu = nu_exact
#d2nudr2 = d2nudr2_exact
#d3nudr3 = d3nudr3_exact

# NFW Density
@jit
def rho_NFW(r, rho_0, r_0):
    x = r/r_0
    return rho_0 * x**(-1) * (1+x)**(-2)

# NFW potential
@jit
def phi_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    x = r/r_0
    return -4*np.pi*G*rho_0*r_0**2 * (np.log(1+x)/x)

# NFW enclosed mass
@jit
def M_enc_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    x = r/r_0
    return 4*np.pi*rho_0*r_0**3 * (np.log(1 + x) - x/(1 + x))

# NFW sigmasq analytic with spence
def sigmasq_NFW(r, rho_0, r_0):
    G = 0.449
    x = r/r_0
    prefactor = 4*np.pi*G*rho_0*r_0**2
    term1 = x*(1+x)**2
    term2 = 1.5*np.log(1+x)**2 + np.log(1+x)/(2*x**2 * (1+x)) - 1.5*np.log(1+x)/(x*(1+x))\
                - 2.5*x*np.log(1+x)/(1+x) - 5.5*np.log(1+x)/(1+x) +2.5*np.log(x) + 1/(2*x*(1+x)) - 1/(2*(1+x))\
                + 3*spence(1+x) + 0.5*np.pi**2 - 3*np.log(x/(1+x)) - 2/(1+x) - 1/(2*(1+x)**2) - 1/x
    return prefactor*term1*term2

# NFW sigmasq line of sight with quadrature
@np.vectorize
def sigmasq_los_NFW(rperp, rho_0, r_0):
    def integrand_num(r):
        return rho_NFW(r, rho_0, r_0)*sigmasq_NFW(r, rho_0, r_0)*r/np.sqrt(r**2 - rperp**2)
    def integrand_denom(r):
        return rho_NFW(r, rho_0, r_0)*r/np.sqrt(r**2 - rperp**2)
    return quad(integrand_num, rperp, np.inf)[0]/quad(integrand_denom, rperp, np.inf)[0]

# NFW circular velocity profile
def vcirc_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return np.sqrt(G*M_enc_NFW(r, rho_0, r_0)/r)

# NFW derivatives needed for integrating to find f(E)
@jit
def dpsidr_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return -G*M_enc_NFW(r, rho_0, r_0)/r**2

@jit
def d2psidr2_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return G*(2*M_enc_NFW(r, rho_0, r_0)/r**3 - (4*np.pi*rho_NFW(r, rho_0, r_0)))

@jit
def drhodr_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    x = r/r_0
    num = 3*x + 1
    denom = x**2 * (1+x)**3
    return -rho_0/r_0 * num/denom

@jit
def d3psidr3_NFW(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return G * (8*np.pi*rho_NFW(r, rho_0, r_0)/r - 3*M_enc_NFW(r, rho_0, r_0)/r**4 - 4*np.pi*drhodr_NFW(r, rho_0, r_0))

#@jit
#def d2rhodr2_NFW(r, R_e, n):
#    rho_0 = 18 # 10^5 solar masses / kpc^3
#    r_0 = 6 # kpc
#    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
#    x = r/r_0
#    num = 2*(6*x**2 + 4*x + 1)
#    denom = x**3 * (1+x)**4
#    return rho_0/r_0**2 * num/denom

#@jit
#def d3rhodr3_NFW(r, R_e, n):
#    rho_0 = 18 # 10^5 solar masses / kpc^3
#    r_0 = 6 # kpc
#    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
#    x = r/r_0
#    num = 6*(10*x**3 + 10*x**2 + 5*x + 1)
#    denom = x**4 * (1+x)**5
#    return -rho_0/r_0**3 * num/denom

@jit
def d2rhodpsi2_NFW(r, rho_0, r_0, R_e, n):
    return dpsidr_NFW(r, rho_0, r_0)**(-2) * (d2nudr2(r, R_e, n) - dnudr(r, R_e, n)*d2psidr2_NFW(r, rho_0, r_0)*(dpsidr_NFW(r, rho_0, r_0))**(-1))
@jit
def d3rhodpsi3_NFW(r, rho_0, r_0, R_e, n):
    term1 = -2*d2psidr2_NFW(r, rho_0, r_0)*(dpsidr_NFW(r, rho_0, r_0))**(-4) * (d2nudr2(r, R_e, n) \
                                                - dnudr(r, R_e, n)*d2psidr2_NFW(r, rho_0, r_0)*dpsidr_NFW(r, rho_0, r_0)**(-1))
    term2 = dpsidr_NFW(r, rho_0, r_0)**(-3) * (d3nudr3(r, R_e, n) \
                               - d2nudr2(r, R_e, n)*d2psidr2_NFW(r, rho_0, r_0)*(dpsidr_NFW(r, rho_0, r_0))**(-1) \
                               + dnudr(r, R_e, n)*(dpsidr_NFW(r, rho_0, r_0))**(-2) * (d2psidr2_NFW(r, rho_0, r_0))**2 \
                               - dnudr(r, R_e, n)*(dpsidr_NFW(r, rho_0, r_0))**(-1) * d3psidr3_NFW(r, rho_0, r_0))
    return term1 + term2

# helpers for calculating dynamical friction timescales
@np.vectorize
def bmax_NFW(r, rho_0, r_0):
    return min(r, np.abs(rho_NFW(r, rho_0, r_0)/drhodr_NFW(r, rho_0, r_0)))

def C_NFW(m, r, v, rho_0, r_0):
    G = 0.449
    x = v/np.sqrt(2*sigmasq_NFW(r, rho_0, r_0))
    maxwellian_terms = erf(x) - 2*x/np.sqrt(np.pi) * np.exp(-x**2)
    coulomb_log = 0.5*np.log(1 + (bmax_NFW(r, rho_0, r_0)/(G*m/v**2))**2)
    return coulomb_log*maxwellian_terms

def tau_NFW(m, r, v, rho_0, r_0):
    G = 0.449
    return v**3 / (4*np.pi*G**2 * rho_NFW(r, rho_0, r_0)*m*C_NFW(m, r, v, rho_0, r_0))

