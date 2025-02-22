import numpy as np


def modified_GC_random_sample(log_Mstellar, n):
    if n % 2 != 0:
        raise ValueError("num_samples must be even for exact 50/50 split.")
    
    # Mean and scatter for log-normal distribution
    Mstellar = 10**log_Mstellar
    mean_log_MGC = np.log10(0.01 * Mstellar)  # Mean in log space
    sigma = 0.5  # dex
    
    # Generate log-normal samples
    MGC_samples = 10**np.random.normal(mean_log_MGC, sigma, n)
    MNSC_samples = 10**np.random.normal(mean_log_MGC, sigma, n)

    # 50/50 split for NSC presence
    half = n // 2
    samples = np.array([0] * half + [1] * half)
    np.random.shuffle(samples)
    
    # Combine GC and NSC samples
    MGCs = []
    for i in range(n):
        if samples[i] == 1:  # Galaxy has an NSC
            MGC = MGC_samples[i] + MNSC_samples[i]
        else:  # Galaxy without NSC
            MGC = MGC_samples[i]
        
        MGCs.append(MGC)

    return np.array(MGCs)