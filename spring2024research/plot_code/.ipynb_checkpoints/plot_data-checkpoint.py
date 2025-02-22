import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
import os
import seaborn as sns
import pandas as pd



def load_files (R_e, GC_mass, halo_mass): #rounded as in txt files
    # open final time file
    file_name0 = 'Sim_NFW_0_'+str(R_e)+'_'+str(GC_mass)+'_'+str(halo_mass)+'.txt'  #initial
    file_name20 = 'Sim_NFW_20_'+str(R_e)+'_'+str(GC_mass)+'_'+str(halo_mass)+'.txt' #final 

    data_initial = np.genfromtxt('/home/vk9342/spring2024research/test_sim_run_1/'+str(file_name0), skip_header=2)  
    data_final = np.genfromtxt('/home/vk9342/spring2024research/test_sim_run_1/'+str(file_name20), skip_header=2)

    iteration_initial = data_initial [:, 0]
    iteration_final = data_final [:, 0]
    all_masses_initial = data_initial [:, 1]     #all initial masses from every iteration for a certain combination params
    all_masses_final = data_final [:, 1]
    return (iteration_initial, iteration_final, all_masses_initial, all_masses_final)


def avg_over_iters (iteration_initial, iteration_final, all_masses_initial, all_masses_final, num_iters):        
    # number of iterations = 10
    # for a given Re, GC mass, Halo mass (two files used - time 0 and time 20)
    # file: row = data for one GC
    # n rows of iteration k: all the GCs in given iteration 
    t = 0
    initial_num_masses = []
    initial_masses = []
    while t<num_iters: #number of iteration 
        for k in range (len(iteration_initial)):
            if iteration_initial[k] == t:        # for a given iteration ie between 0 and 9
                initial_masses.append(all_masses_initial[k])   #clears at the end of every t value
        # initial_masses here = all GC masses from given iteration
        initial_num_masses.append(len(initial_masses)) #number of GCs for given iteration t
        initial_masses = []  #clear initial masses
        t = t+1
    # while loop returns array where each value is the initial number of GCs in an iteration 
    # shape = array of length number of iterations, values = number of GCs)

    #does same for final masses ie the time = 20 file 
    t = 0
    final_masses = []
    final_num_masses = []
    max_masses = []
    while t<num_iters:
        for k in range (len(iteration_final)):
            if iteration_final[k] == t:
                final_masses.append (all_masses_final[k])   
        final_num_masses.append(len(final_masses)) 
        max_masses.append(max(final_masses))
        final_masses = []
        t = t+1

    max_mass_final = sum(max_masses)/len(max_masses)  #max mass averaged over all iterations
    num_mergers = [] #number of mergers
    for i in range (num_iters):
        num_mergers.append (initial_num_masses[i] - final_num_masses[i])  #array of length num_iterations; values = num of mergers in given iteration
    avg_num_mergers = sum(num_mergers)/len(num_mergers)  #average the number of mergers that have occured over all iterations

    return (avg_num_mergers)

import itertools
def does_have_merger (R_es, GC_masses, halo_masses, num_iters):
    import itertools
    for R_e, GC_mass, halo_mass in itertools.product(R_es, GC_masses, halo_masses): #iterate through every combo of Re, total GC mass, DM halo mass
    
        no_merger_Re = []
        no_merger_GC = []
        no_merger_halo = []
    
        merger_Re = []
        merger_GC = []
        merger_halo = []
        
        merger_num = []
        merger_num_plot = []    
    
        iteration_initial, iteration_final, all_masses_initial, all_masses_final = load_files(
            R_e, GC_mass, halo_mass)
        
        avg_num_mergers = avg_over_iters (iteration_initial, iteration_final, 
                                      all_masses_initial, all_masses_final, num_iters)
        
        #add simulation params to no merger and merger arrays     
        if avg_num_mergers <=1:
            no_merger_Re.append(R_e)
            no_merger_GC.append(GC_mass)
            no_merger_halo.append(halo_mass)
    
        else:
            merger_Re.append(R_e)
            merger_GC.append(GC_mass)
            merger_halo.append(halo_mass)
            merger_num.append(avg_num_mergers)


        
    return (merger_Re, no_merger_Re, merger_GC, no_merger_GC, merger_halo, no_merger_halo)