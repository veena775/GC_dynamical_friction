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
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'


#### DEFINE GAUSSIAN FUNCTIONS
## Gaussian function
def Gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

## Get counts and bincenters from histogram (bincenters later to be used as x values for Gaussian)
def Gaussian_counts_bincenters(x, bins):
    counts, bin_edges = np.histogram(x, bins=bins, density = True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    return [counts, bin_centers]
   
## Get Gaussian params - mu and sigma
def Gaussian_params(x, bins):
    counts, bin_centers = Gaussian_counts_bincenters(x, bins)
    p0 = [np.mean(x), np.std(x)]  # Initial guess for the parameters
    params, cov = curve_fit(Gaussian, bin_centers, counts, p0=p0)
    return params 


################################################
## DARK MATTER HALO MASS
################################################
# ELVES. IV. The Satellite Stellar-to-halo Mass Relation Beyond the Milky Way
# Values and equation from Shany's paper - section 4.2, eq 1
# Equation from Shany's paper

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


################################################
## Re
################################################
# Structures of Dwarf Satellites of Milky Way-like Galaxies: Morphology, Scaling Relations, and Intrinsic Shapes
# Eq 5 section 3.5

def logRe_eq (logMstellar):
    a = 1.077
    b = 0.246
    logRe = a + b*logMstellar
    return logRe
sigmaRe = 0.181


################################################
## GC MASS
################################################
## data that Shany sent on slack  
## from Clearing the Hurdle: The Mass of Globular Cluster Systems as a Function of Host Galaxy Mass
## Gwendolyn M. Eadie et al 2022 ApJ 926 162

dataGC = np.genfromtxt('best_sample.txt', skip_header=1, delimiter=',')
M_GC = dataGC[:, 2] # log GC Mass (in solar masses)
M_S = dataGC[:, 4] #log Stellar Mass

### choose a range of stellar masses to isolate a vertical 'stripe from data'
def Stripe_bounds(MStellar):
    if 8.5 <= MStellar <= 9:
        lbound = 8.5
        ubound = 9
        histbins = 10
    elif 9 < MStellar <= 10.5:
        lbound = 9
        ubound = 10.5
        histbins = 12 #??
    elif 7 <= MStellar < 8.5:
        lbound = 7
        ubound = 10.5
        histbins = 12
    return(lbound, ubound, histbins)

## get stellar and GC masses within stripe
def GC_stripe_func (M_GC, M_S, lbound, ubound):
    M_GCstripe = [] 
    M_Sstripe = []
    n = len(M_GC)
    for i in range (n):
        if M_S[i] >= lbound and M_S[i] <= ubound and M_GC[i]!=0:
            M_GCstripe.append(M_GC[i])
            M_Sstripe.append(M_S[i])
    return [M_GCstripe, M_Sstripe]

#####################   SAMPLING MASSES FROM DISTRIBUTION ############################    
##sampling GC Mass for a given Stellar Mass
def GC_random_sample (M_GCstripe, n):
    mu, sigma = Gaussian_params(M_GCstripe, bins=10)
    sample_masses = np.random.normal(mu, sigma, n)
    return sample_masses


############## SAMPLE MASSES FROM DISTRIBUTION ##############################
##### Sampling DM Halo mass

def DM_random_sample (logMstellar, n):
    mu = logMhalo_eq(logMstellar)  ##log
    sigma = std_eq(mu)  ##log
    sample_masses = np.random.normal(mu, sigma, n)
    return (sample_masses)
 
#####################   SAMPLING MASSES FROM DISTRIBUTION ############################    
##sampling GC Mass for a given Stellar Mass
def Re_random_sample (logMstellar, n):
    mu = logRe_eq(logMstellar)
    sigma = 0.181  
    sample_Re = np.random.normal(mu, sigma, n)
    return sample_Re