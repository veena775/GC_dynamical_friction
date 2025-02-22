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