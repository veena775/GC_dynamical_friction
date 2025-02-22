import numpy as np
import math
from ParameterDistributions import *

#So rho_0 and r_0 are going to depend on the mass and concentration of the halo, denoted by c
#get concentrations from mass is Diemer & Joyce 2019  #https://iopscience.iop.org/article/10.3847/1538-4357/aafad6

# a fitting function to Diemer & Joyce 2019 z = 2 c(M):
#it’s a “piecewise-power-law fit to the concentration-mass relation at z = 2 from Diemer & Joyce (2019)”

def c_func(M_halo):  #in units of solar mass
    c_values = np.array([5.606078101846313, 5.505601101255639, 5.296315905448495, 5.094986297483347, 4.901309860470804,     4.666519328282937, 4.500743526125894, 4.329656123893253, 4.154325163981534, 3.975808982640419, 3.834570325527022, 3.727126007769686, 3.6040211391832533, 3.512099050898484, 3.4225214744765236, 3.4402523213171285])

    scaled_M_values = np.array([1208795796.6963477, 1628413544.0132532, 2799360456.643171, 5080218046.9130125, 8973072494.285637, 18147780986.393196, 31197345819.126064, 52197182204.356415, 97327438615.81503, 191581222925.96384, 347677407476.5763, 566162599282.2716, 1114445470753.5625, 2193696137571.7903, 5976825664072.4, 14221361511653.348])
    scaled_M = M_halo/0.7
    
    return 10**np.interp(np.log10(scaled_M), np.log10(scaled_M_values), np.log10(c_values))


def rho_0_func (M_halo):
    rho_crit = 0.00136 # 10^5 Solar masses / kpc^3
    c = c_func(M_halo)
    rho_0 = 200/3 * c**3 * rho_crit / (np.log(1+c) - c/(1+c))
    return (rho_0)   # in 10^5 solar masses / kpc^3 ?????

def r_0_func (M_halo):
    rho_crit = 136 #Solar masses / kpc^3
    c = c_func(M_halo)
    r_0 = 1/c * (3*M_halo/ (800 * math.pi * rho_crit))**(1/3)
    return (r_0)  ## in kpc