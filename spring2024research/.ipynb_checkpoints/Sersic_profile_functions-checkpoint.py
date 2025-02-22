import numpy as np
from math import gamma
from numba import jit
from scipy.integrate import quad

"""
# Normalized Sersic(r), unnormalized exact deprojection density, and its derivatives
@jit
def Sersic(r, R_e, n):
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(15515*n**2) + 131/(1148175*n**3) - 2194697/(306090717750*n**4)
    x = (r/R_e)**(1/n)
    return b_n**n / (n*gamma(n)) * np.exp(-b_n*x)

def nu_exact(r, R_e, n):
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(15515*n**2) + 131/(1148175*n**3) - 2194697/(306090717750*n**4)
    x = r/R_e
    integrand = lambda t: np.exp(-b_n*t**(1/n))*(t**(1/n - 1))/np.sqrt(t**2 - x**2)
    return quad(integrand, r/R_e, np.inf)[0]

def dnudr_exact(r, R_e, n):
    forward_step = r+1e-8*r # roughly floating point error size
    difference = nu_exact(forward_step, R_e, n) - nu_exact(r, R_e, n)
    return difference/(forward_step-r)

def d2nudr2_exact(r, R_e, n):
    forward_step = r+1e-8*r # roughly floating point error size
    difference = dnudr_exact(forward_step, R_e, n) - dnudr_exact(r, R_e, n)
    return difference/(forward_step-r)

def d3nudr3_exact(r, R_e, n):
    forward_step = r+1e-8*r # roughly floating point error size
    difference = d2nudr2_exact(forward_step, R_e, n) - d2nudr2_exact(r, R_e, n)
    return difference/(forward_step-r)
"""

# Prugniel Simien Approximate density nu(r) that projects to a Sersic, unnormalized
@jit
def nu(r, R_e, n):
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(15515*n**2) + 131/(1148175*n**3) - 2194697/(306090717750*n**4)
    p_n = 1 - 0.6097/n + 0.05463/n**2
    x = r/R_e
    return x**(-p_n) * np.exp(-b_n*x**(1/n))

# derivatives of Prugniel Simien nu(r):
@jit
def dnudr(r, R_e, n):
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(15515*n**2) + 131/(1148175*n**3) - 2194697/(306090717750*n**4)
    p_n = 1 - 0.6097/n + 0.05463/n**2
    x = r/R_e
    polynomial = b_n*x**(1/n) + n*p_n
    denom = n*r
    return -polynomial/denom * np.exp(-b_n*x**(1/n))*x**(-p_n)

@jit
def d2nudr2(r, R_e, n):
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(15515*n**2) + 131/(1148175*n**3) - 2194697/(306090717750*n**4)
    p_n = 1 - 0.6097/n + 0.05463/n**2
    x = r/R_e
    polynomial = b_n**2 * x**(1/n) + (2*n*p_n + n - 1)*b_n*x**(1/n) + n**2 * p_n**2 + n**2 * p_n
    denom = (n*r)**2
    return polynomial/denom * np.exp(-b_n*x**(1/n))*x**(-p_n)

@jit
def d3nudr3(r, R_e, n):
    b_n = 2*n - 1/3 + 4/(405*n) + 46/(15515*n**2) + 131/(1148175*n**3) - 2194697/(306090717750*n**4)
    p_n = 1 - 0.6097/n + 0.05463/n**2
    x = r/R_e
    polynomial1 = b_n**3 * x**(3/n)
    polynomial2 = (3*n*p_n + 3*n - 3)*b_n**2 * x**(2/n)
    polynomial3 =  (3*n**2 * p_n**2 + (6*n**2 - 3*n)*p_n + 2*n**2 - 3*n + 1) * b_n*x**(1/n)
    polynomial4 = n**3 * p_n**3 + 3*n**3 * p_n**2 + 2*n**3 * p_n
    polynomial = polynomial1 + polynomial2 + polynomial3 + polynomial4
    denom = (n*r)**3
    return -polynomial/denom * np.exp(-b_n*x**(1/n))*x**(-p_n)