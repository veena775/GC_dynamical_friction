import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.stats import norm
import math
import statistics
from scipy.optimize import fsolve
import matplotlib as mpl
from gaussian_functions import *


#####################
##FIND M DARK MATTER HALO VALUE FROM M STELLAR

#ELVES. IV. The Satellite Stellar-to-halo Mass Relation Beyond the Milky Way
# Values and equation from Shany's paper - section 4.2, eq 1

## Equation from Shany's paper
def logMstellar_eq(logMhalo):
    M1 = 11.889
    alpha = 2.1
    beta = 0.464
    epsilon = -1.432
    loggamma = -0.812
    gamma = 10**loggamma
    delta = 0.319
    x = np.log10(10**logMhalo/10**M1)
    logMstellar = epsilon + M1 - np.log10(10**(-alpha*x) + 10**(-beta*x)) + gamma*np.exp(-0.5*(x/delta)**2)
    return logMstellar

## Solve for Mhalo value for given Mstellar value (solve y(x) eq for x value)
def logMhalo_eq (logMstellar):
    func = lambda x: logMstellar_eq(x) - logMstellar  ## def f(x) = 0
    logMhalo_guess = 12
    logMhalo = fsolve(func, logMhalo_guess) ##returns roots of func
    return(logMhalo)


### STNDARD DEVIATION
def std_eq (logMhalo):
    sigma0 = 0.02
    nu = -0.47
    M1 = 11.889
    logsigma = sigma0 + nu * (logMhalo - M1)
    return (logsigma)

def DM_random_sample (logMstellar, n):
    mu = logMhalo_eq(logMstellar)  ##log
    sigma = std_eq(mu)  ##log
    logsample_masses = np.random.normal(mu, sigma, n)
    sample_masses  = 10**logsample_masses
    return (sample_masses) 


def c_func(M_halo): #in solar masses
    c_values = np.array([5.606078101846313, 5.505601101255639, 5.296315905448495, 
                         5.094986297483347, 4.901309860470804, 4.666519328282937, 
                         4.500743526125894, 4.329656123893253, 4.154325163981534, 
                         3.975808982640419, 3.834570325527022, 3.727126007769686, 
                         3.6040211391832533, 3.512099050898484, 3.4225214744765236, 
                         3.4402523213171285])
    scaled_M_values = np.array([1208795796.6963477, 1628413544.0132532, 
                                2799360456.643171, 5080218046.9130125, 
                                8973072494.285637, 18147780986.393196, 
                                31197345819.126064, 52197182204.356415, 
                                97327438615.81503, 191581222925.96384, 
                                347677407476.5763, 566162599282.2716, 
                                1114445470753.5625, 2193696137571.7903, 
                                5976825664072.4, 14221361511653.348])
    scaled_M = M_halo/0.7
    return 10**np.interp(np.log10(scaled_M), np.log10(scaled_M_values), np.log10(c_values))


# NFW halo - eqs (3) and (4)
# Mstellar = 1.72*10**8
# logMstellar = np.log10(Mstellar)
# M_halo = 10**logMhalo_eq(logMstellar)
# print(M_halo)
# M_halo = 106210089323.59808 
# M_halo = 10**10.34446449

def rho_0_func (M_halo):#in solar masses 
    rho_crit = 0.00136 
    c = c_func(M_halo)
    rho_0 = 200/3 * c**3 * rho_crit / (np.log(1+c) - c/(1+c))
    return (rho_0)

def r_0_func (M_halo):  #in solar masses 
    M_halo = M_halo*10**(-5)
    rho_crit = 0.00136
    c = c_func(M_halo)
    r_0 = 1/c * (3*M_halo/ (800 * math.pi * rho_crit))**(1/3)
    return (r_0)