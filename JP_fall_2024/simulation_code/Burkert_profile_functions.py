import numpy as np
from numba import jit
from Sersic_profile_functions import *
from scipy.integrate import quad
from scipy.special import gamma, erf, spence

#nu = nu_exact
#d2nudr2 = d2nudr2_exact
#d3nudr3 = d3nudr3_exact

# Burkert density
@jit
def rho_Burkert(r, rho_0, r_0):
    return rho_0*r_0**3 / ((r_0 + r) * (r_0**2 + r**2))

# Burkert potential
@jit
def phi_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    x = r/r_0
    return np.pi * G * rho_0 * r_0**2 * \
            ((1 - 1/x)*np.log(1 + x**2) - 2*(1 + 1/x)*np.log(1 + x) + 2*(1 + 1/x)*np.arctan(x) - np.pi)

# Burkert enclosed mass
@jit
def M_enc_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    x = r/r_0
    return np.pi*rho_0*r_0**3 * (np.log(1 + x**2) + 2*np.log(1 + x) - 2*np.arctan(x))

# Burkert sigmasq, no convenient analytic form, have to use quad
@np.vectorize
def sigmasq_Burkert(r, rho_0, r_0):
    G = 0.449
    x = r/r_0
    prefactor = np.pi*G*rho_0*r_0**2
    term1 = (1+x)*(1+x**2)
    integrand = lambda u: (np.log(1+u**2)+2*np.log(1+u)-2*np.arctan(u))/(u**2 * (1+u)*(1+u**2))
    term2 = quad(integrand, x, np.inf)[0]
    return prefactor*term1*term2

# Burkert sigmasq line of sight with quadrature, approximating the integrand with an interpolator for speed
@np.vectorize
def sigmasq_los_Burkert(rperp, rho_0, r_0):
    r_values = np.logspace(-3, 5, 100)
    sigmasq_values = sigmasq_Burkert(r_values, rho_0, r_0)
    def sigmasq_interp(r):
        return 10**np.interp(r, r_values, np.log10(sigmasq_values))
    def integrand_num(r):
        return rho_Burkert(r, rho_0, r_0)*sigmasq_interp(r)*r/np.sqrt(r**2 - rperp**2)
    def integrand_denom(r):
        return rho_Burkert(r, rho_0, r_0)*r/np.sqrt(r**2 - rperp**2)
    return quad(integrand_num, rperp, 2e4)[0]/quad(integrand_denom, rperp, 2e4)[0]

# Burkert circular velocity
def vcirc_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return np.sqrt(G*M_enc_Burkert(r, rho_0, r_0)/r)

# Burkert derivatives needed for integrating to find f(E)
@jit
def dpsidr_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return -G*M_enc_Burkert(r, rho_0, r_0)/r**2

@jit
def d2psidr2_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return G*(2*M_enc_Burkert(r, rho_0, r_0)/r**3 - (4*np.pi*rho_Burkert(r, rho_0, r_0)))

@jit
def drhodr_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    x = r/r_0
    num = 3*x**2 + 2*x + 1
    denom = (1 + x)**2 * (1 + x**2)**2
    return -rho_0/r_0 * num/denom

@jit
def d3psidr3_Burkert(r, rho_0, r_0):
    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
    return G * (8*np.pi*rho_Burkert(r, rho_0, r_0)/r - 3*M_enc_Burkert(r, rho_0, r_0)/r**4 - 4*np.pi*drhodr_Burkert(r, rho_0, r_0))

#@jit
#def d2rhodr2_Burkert(r):
#    rho_0 = 166 # 10^5 solar masses / kpc^3
#    r_0 = 2 # kpc
#    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
#    x = r/r_0
#    num = x**2 * (3*x**2 + 4*x + 3)
#    denom = (1 + x)**3 * (1 + x**2)**3
#    return 4*rho_0/r_0**2 * num/denom

#@jit
#def d3rhodr3_Burkert(r):
#    rho_0 = 166 # 10^5 solar masses / kpc^3
#    r_0 = 2 # kpc
#    G = 0.449 # (kpc/Gyr)^2 * kpc / 10^5 solar masses
#    x = r/r_0
#    num = x*(5*x**5 + 10*x**4 + 10*x**3 - 3*x - 2)
#    denom = (1 + x)**4 * (1 + x**2)**4
#    return -12*rho_0/r_0**3 * num/denom

@jit
def d2rhodpsi2_Burkert(r, rho_0, r_0, R_e, n):
    return dpsidr_Burkert(r, rho_0, r_0)**(-2) * (d2nudr2(r, R_e, n) - dnudr(r, R_e, n)*d2psidr2_Burkert(r, rho_0, r_0)*(dpsidr_Burkert(r, rho_0, r_0))**(-1))
@jit
def d3rhodpsi3_Burkert(r, rho_0, r_0, R_e, n):
    term1 = -2*d2psidr2_Burkert(r, rho_0, r_0)*(dpsidr_Burkert(r, rho_0, r_0))**(-4) * (d2nudr2(r, R_e, n) \
                                                - dnudr(r, R_e, n)*d2psidr2_Burkert(r, rho_0, r_0)*dpsidr_Burkert(r, rho_0, r_0)**(-1))
    term2 = dpsidr_Burkert(r, rho_0, r_0)**(-3) * (d3nudr3(r, R_e, n) \
                               - d2nudr2(r, R_e, n)*d2psidr2_Burkert(r, rho_0, r_0)*(dpsidr_Burkert(r, rho_0, r_0))**(-1) \
                               + dnudr(r, R_e, n)*(dpsidr_Burkert(r, rho_0, r_0))**(-2) * (d2psidr2_Burkert(r, rho_0, r_0))**2 \
                               - dnudr(r, R_e, n)*(dpsidr_Burkert(r, rho_0, r_0))**(-1) * d3psidr3_Burkert(r, rho_0, r_0))
    return term1 + term2

# helpers for calculating dynamical friction timescales
@np.vectorize
def bmax_Burkert(r, rho_0, r_0):
    return min(r, np.abs(rho_Burkert(r, rho_0, r_0)/drhodr_Burkert(r, rho_0, r_0)))

def C_Burkert(m, r, v, rho_0, r_0):
    G = 0.449
    x = v/np.sqrt(2*sigmasq_Burkert(r, rho_0, r_0))
    maxwellian_terms = erf(x) - 2*x/np.sqrt(np.pi) * np.exp(-x**2)
    coulomb_log = 0.5*np.log(1 + (bmax_Burkert(r, rho_0, r_0)/(G*m/v**2))**2)
    return coulomb_log*maxwellian_terms

def tau_Burkert(m, r, v, rho_0, r_0):
    G = 0.449
    return v**3 / (4*np.pi*G**2 * rho_Burkert(r, rho_0, r_0)*m*C_Burkert(m, r, v, rho_0, r_0))
