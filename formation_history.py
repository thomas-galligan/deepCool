import pynbody
import matplotlib.pyplot as plt
import numpy as np
font={'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
plt.rc('text',usetex=True)
import gc
gc.collect()


def correct_mass(A_in):
    """ this function rounds the stellar masses up to the nearest thousand,
        to correct for the effect of supernovae. It takes in two arrays: final stellar
        mass, and corresponding time of formation """
    
    tform = A_in[:,0]
    mass = A_in[:,1]
   
    for i in range(len(mass)):
        if mass[i]<1000 and tform[i]>50:
                mass[i] = 1000
        elif mass[i]<2000:
                mass[i] = 2000
        elif mass[i]<3000:
                mass[i] = 3000
        else:
             	mass[i] = 4000

    A = np.zeros((len(mass),2))
    A[:,0] = tform
    A[:,1] = mass
    return A


#------Old RAMSES sim----------
s_old = pynbody.load('/mnt/extraspace/tpgalligan/ramses/runs/rt_with_sn/RAMSES_simulation/output_00035')
#s_old['pos']-=s_old.properties['boxsize']/2.0
s_old.physical_units()
A_old = np.zeros((len(s_old.s['mass']),2))
A_old[:,0] = s_old.s['tform'].in_units('Myr')
A_old[:,1] = s_old.s['mass'].in_units('Msol')
A_old = correct_mass(A_old)
dt = 10 # size of bin in Myr
A_old = A_old[A_old[:,0].argsort()] # sort A by first column
rate_old = []
for i in range(0,500,dt):
    # sum the masses of stars formed in the range in question    
    mask_old = (A_old[:,0]>i)*(A_old[:,0]<(i+dt))
    m_tot_old = (A_old[:,1]*mask_old).sum()
    rate_old.append(m_tot_old/(dt*1.0e6))



#---------New RAMSES sim--------
s_new = pynbody.load('/mnt/extraspace/tpgalligan/ramses/runs/DEEP_METAL_RAMSES/RAMSES_simulation/output_00035')
#s_new['pos']-=s_new.properties['boxsize']/2.0
s_new.physical_units()
A_new = np.zeros((len(s_new.s['mass']),2))
A_new[:,0] = s_new.s['tform'].in_units('Myr')
A_new[:,1] = s_new.s['mass'].in_units('Msol')
A_new = correct_mass(A_new)
dt = 10 # size of bin in Myr
A_new = A_new[A_new[:,0].argsort()] # sort A by first column
rate_new = []
for i in range(0,500,dt):
    # sum the masses of stars formed in the range in question    
    mask_new = (A_new[:,0]>i)*(A_new[:,0]<(i+dt))
    m_tot_new = (A_new[:,1]*mask_new).sum()
    rate_new.append(m_tot_new/(dt*1.0e6))



#------plotting--------

t = np.linspace(dt,500,50)    

plt.tick_params(direction='in')
plt.semilogy(t,rate_old,color='black',label='old RAMSES')
plt.semilogy(t,rate_new,color='blue',label='RAMSES with deepMetal')
plt.xlabel(r'\rm{Age of the universe [Myr]}')
plt.ylabel(r'\rm{SFR} [M$_{\odot}\rm{yr}^{-1}]$')
plt.legend()
plt.show()

