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
from gaussian_functions import *

def logRe_eq (logMstellar):
    a = 1.028
    b = 0.259
    logRe = a + b*logMstellar
    if logMstellar > 8.3:
        logRe = a + b*8.3
    return logRe  #in pc

def Re_random_sample (logMstellar, n):
    mu = logRe_eq(logMstellar)
    sigma = 0.152
    logsample_Re = np.random.normal(mu, sigma, n) #in pc
    # sample_Re = logsample_Re
    sample_Re = 10**logsample_Re
    return sample_Re #in pc