halo_masses =    ##log value 
for i in range(len(halo_masses)):
    halo_masses[i] = 10**halo_masses[i]
  
rho_0 = []
r_0 = []
for i in range(len(halo_masses)):
    rho_0.append(rho_0_func(halo_masses[i]))
    r_0.append (r_0_func(halo_masses[i]))

print('rho: ', rho_0)
print('r: ', r_0)