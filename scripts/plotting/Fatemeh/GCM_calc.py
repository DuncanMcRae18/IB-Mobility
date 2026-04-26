#Description: Computes IB energy levels for desired GCM (global current mismatch), by Fatemeh MK
#Units are in electron Volts

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy import integrate
import scipy.optimize as optimize

h = const.physical_constants["Planck constant in eV s"][0]
c = const.c 
k = const.physical_constants["Boltzmann constant in eV/K"][0]
sigma = const.Stefan_Boltzmann 
q =  const.elementary_charge 
PI = math.pi

Ts = 6000   #solar temp
Eg = 2.5
GCM_percent = [0, 10, 25, 50]

def photon_flux_integrand(E, T, mu):
    if (mu>E):
        print("OH NO")
        print(mu, E)   
    fun = math.pow(E, 2)/(np.exp((E - mu)/(k*T)) - 1)
    return 2*PI*fun/(math.pow(h,3)*math.pow(c,2))

def N_IV_abs(Ei, Eg):
   #Photon absorption from VB to the IB
    Ec = Eg - Ei
    Eup = Ec  
    if Ei > Eg/2:   #if the IB is in the upper half of the bandgap
        Eup = Eg
    N_IV = integrate.quad(photon_flux_integrand,Ei, Eup, args=(Ts,0))[0]
    return N_IV

def N_CI_abs(Ei, Eg):
    #Photon absorption from IB to the CB
    Ec = Eg - Ei
    Eup = Eg
    if Ei > Eg/2:   #if the IB is in the upper half of the bandgap
        Eup = Ei
    N_CI = integrate.quad(photon_flux_integrand,Ec, Eup, args=(Ts,0))[0]
    return N_CI

def GCM(Ei, Eg):    #calculates the global current mismatch as defined in the 2020 conference paper
    num = abs(N_CI_abs(Ei, Eg) - N_IV_abs(Ei, Eg))
    denom = N_CI_abs(Ei, Eg) + N_IV_abs(Ei, Eg)
    m_G = num/denom*100  #%
    return m_G

def EqToSolve(Ei, Eg, GCM_desired):
    return (GCM(Ei, Eg) - GCM_desired)

IB_E = [] 
for gcm in GCM_percent:
    x = optimize.fsolve(EqToSolve, Eg/3, args=(Eg, gcm))[0]
    IB_E.append(x)

print(IB_E)

'''
#Plotting EqToSolve() to visually see the solutions
y = []
x = np.linspace(0.1, 2.4, 200)
for xx in x:
    y.append(EqToSolve(xx, Eg, GCM_percent[1]))

plt.plot(x, y), plt.xlabel('E_I [eV]'), plt.ylabel('GCM(E_i) - desired_GCM')
plt.title("IB levels that give the desired GCM")
plt.axhline(y = 0, color ='k')
plt.show()
'''

'''
#Plotting the GCM as a function of E_I
y = []
x = np.linspace(0.1, 2.4, 200)
for xx in x:
    y.append(GCM(xx, Eg))

plt.plot(x, y), plt.xlabel('E_I [eV]'), plt.ylabel('GCM [%]')
plt.title(f"Global Current Mismatch vs IB energy level for Eg = {Eg:.1f} eV ")
plt.axhline(y = 0, color ='k', linestyle='-')
plt.text(0.1, 10, "10%")
plt.axhline(y = 10, color ='k', linestyle='--')
plt.text(0.1, 25, "25%")
plt.axhline(y = 25, color ='k', linestyle='--')
plt.text(0.1, 50, "50%")
plt.axhline(y = 50, color ='k', linestyle='--')
plt.show()
'''

# For Eg = 1.67 eV,
# IB_E values:
# (0%GCM): 0.58974918 or 1.08025082 eV
# (10%GCM): 0.55447173 or 1.04770392 eV
# (25%GCM): 0.49546686  or 1.00324958 eV
# (50%GCM): 0.37576187 or 0.93862021 eV

# For Eg = 2.5 eV,
# IB_E values:
# (0%GCM): 0.9739149949131315 
# (10%GCM): 0.9327205938861218 
# (25%GCM):  0.8622691192926665 
# (50%GCM): 0.710701104724266 


