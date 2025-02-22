import numpy as np
import math
from ParameterDistributions import *

logMstellar = 8.5
n = 20

DM_sample = DM_random_sample(logMstellar, n)  #log
Re_sample = Re_random_sample(logMstellar, n)  #log 

lbound, ubound, histbins = Stripe_bounds(logMstellar)
M_GCstripe, M_Sstripe = GC_stripe_func (M_GC, M_S, lbound, ubound)
GC_sample = GC_random_sample(M_GCstripe, n)  #log 


print ('Re: ', Re_sample) 
print ('GC: ', GC_sample)
print ('Halo: ', DM_sample)