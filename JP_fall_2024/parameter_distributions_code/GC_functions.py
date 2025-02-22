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
##plt.rcParams['text.usetex'] = False
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
from gaussian_functions import *



## data that Shany sent on slack  
## from Clearing the Hurdle: The Mass of Globular Cluster Systems as a Function of Host Galaxy Mass
## Gwendolyn M. Eadie et al 2022 ApJ 926 162

def GC_stripe_func (logMStellar):
    if 8.5 <= logMStellar <= 9:
        lbound = 8.5
        ubound = 9
        histbins = 10
    elif 9 < logMStellar <= 10.5:
        lbound = 9
        ubound = 10.5
        histbins = 12 #??
    elif 7 <= logMStellar < 8.5:
        lbound = 7
        ubound = 8.5
        histbins = 12

    dataGC = np.genfromtxt('best_sample.txt', skip_header=1, delimiter=',')
    logM_GC = dataGC[:, 2] # log GC Mass (in solar masses)
    logM_S = dataGC[:, 4] #log Stellar Mass
    logM_GCstripe = [] 
    logM_Sstripe = []
    n = len(logM_GC)
    for i in range (n):
        if logM_S[i] >= lbound and logM_S[i] <= ubound and logM_GC[i]!=0:
            logM_GCstripe.append(logM_GC[i])
            logM_Sstripe.append(logM_S[i])
    return [logM_GCstripe, logM_Sstripe, histbins]


def GC_random_sample (logMStellar, n):
    logM_GCstripe, logM_Sstripe, histbins = GC_stripe_func (logMStellar)
    mu, sigma = Gaussian_params(logM_GCstripe, bins=10)
    logsample_masses = np.random.normal(mu, sigma, n)
    sample_masses = 10**logsample_masses
    return sample_masses

