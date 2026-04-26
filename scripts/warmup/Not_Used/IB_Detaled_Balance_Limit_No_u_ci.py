# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, minimize, root
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np
import pandas as pd
import warnings
import traceback
# ThreadPoolExecutor removed - no longer using parallelization

# simplify unit registry
u = UnitRegistry()

# simplify constants
h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt
pi = np.pi

T_sun = 6000 * u.kelvin # temperature of the sun
T_cell = 300 * u.kelvin # temperature of the solar cell


"""This python code is intended to model the detailed balance limit of efficiency for an intermediate band solar cell
   The functions are as follows:
        photon_flux - calculates the blackbody photon flux given some bandgap E_g, temperature of the material, and the chemical potential of that material
        power_input - calculates the total power input from the sun
        IB_current_density - calculates the current density for the intermediate band
        CB_current_density - calculates the current density for the conduction band
        find_u_ci - finds the chemical potential between the intermediate and conduction bands
        rough_efficiency - finds a close approximation to the maximum efficiency for each bandgap energy
        efficiency - calculates the efficiency for a given bandgap energy and chemical potential
        optimize - optimizes the chemical potential to find the maximum efficiency for each bandgap energy
        main - main function to run the calculations and optionally plot and export results
        plot_values - plots the efficiency vs intermediate bandgap energy
        export - exports the results to a CSV file"""

def power_input(T_sun, cutoff, fX):
    # remove units from constants
    T_sun = T_sun.to('K').magnitude
    k = u.Quantity('boltzmann_constant').m_as('eV/K')

    # integrate over blackbody spectrum
    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0, cutoff)

    # calculate power input
    P_in = fX * (pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4')

    # return power
    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)

    N = (fX * pi / ((h**3) * (c**2))) * flux * eV**3

    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    u_iv = u_cv - u_ci

    if E_i < E_c:
        flux_val_abs = photon_flux(E_i, E_c, T_sun, 0 * eV, fX)
        flux_val_emit = photon_flux(E_i, E_c, T_cell, u_iv, 1)
        flux_conduct_abs = photon_flux(E_c, E_g, T_sun, 0 * eV, fX)
        flux_conduct_emit = photon_flux(E_c, E_g, T_cell, u_ci, 1)
    else:
        flux_val_abs = photon_flux(E_i, E_g, T_sun, 0 * eV, fX)
        flux_val_emit = photon_flux(E_i, E_g, T_cell, u_iv, 1)
        flux_conduct_abs = photon_flux(E_c, E_i, T_sun, 0 * eV, fX)
        flux_conduct_emit = photon_flux(E_c, E_i, T_cell, u_ci, 1)


    flux = (flux_conduct_abs - flux_conduct_emit) + (flux_val_emit - flux_val_abs)

    J = q * flux

    return J

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv, rough):

    CB_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv, 1)

    if E_i  <= E_c:
        IB_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci, 1)
        IB_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv, 1)

    else:
        IB_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci, 1) 
        IB_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv, 1)

    if rough:
        IB_flux = min(IB_flux_C, IB_flux_V)
    else:
        IB_flux = IB_flux_C

    flux = CB_flux + IB_flux

    J = q * flux

    # return current density
    return J

def efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in):

    # simopify constants and variables
    E_i = x[0] * eV
    E_c = E_g - E_i
    u_cv = x[1] * eV
    u_ci = x[2] * eV
    u_iv = u_cv - u_ci

    # check to ensure band gaps are valid
    if E_i <= 0 or E_c <= 0:  
        return 0

    # check to ensure potentials are valid
    if not (0 * eV < u_cv < E_g) or not (0 * eV < u_ci < min(E_c,u_cv)) or not (0 * eV < u_iv < min(E_i,u_cv)): 
        return 0        

    # calculate output power and efficiency
    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv, True)
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    # ensure that efficiency is within physical bounds
    if not 0 < efficiency.magnitude < 1:  # Theoretical max ~68% for ideal solar cells
        return 0
    
    # return negative efficiency
    return -efficiency.magnitude

def Optimize(E_g, int_max, T_cell, T_sun, fX):
    # Calculate power input once
    P_in = power_input(T_sun, int_max, fX)

    E_g_val = E_g.m_as('eV')

    bounds = [
        (0.01 * E_g_val, 0.99 * E_g_val),   # E_i bounds
        (0.01 * E_g_val, 0.99 * E_g_val),   # u_cv bounds
        (0.01 * E_g_val, 0.99 * E_g_val),   # u_ci bounds
    ]

    guess = differential_evolution(
        lambda x: efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in),
        bounds=bounds, strategy='best1bin', maxiter=500, popsize=20, tol=1e-3, polish=False)

    result = minimize(efficiency, guess.x,args=(E_g, T_cell, T_sun, int_max, fX, P_in), bounds=bounds, tol=1e-8, method='SLSQP')

    if result.success and result.fun < 0:
        print(f"Optimization successful for E_g={E_g.m_as('eV')}: Efficiency={-result.fun*100:.8f}%, E_i={result.x[0]:.8f} eV, u_cv={result.x[1]:.8f} eV")
        return result.fun, result.x[0]
    elif guess.success and guess.fun < 0:
        print(f"Optimization fell back to rough guess for E_g={E_g.m_as('eV')}: Efficiency={-guess.fun*100:.8f}%, E_i={guess.x[0]:.8f} eV, u_cv={guess.x[1]:.8f} eV")
        return guess.fun, guess.x[0]   
    else:
        print(f"Optimization failed for E_g={E_g.m_as('eV')}")
        return 0, 0

"""This portion of this code defines the parameters you wish to use"""

cplx = 50
int_max = 10

E_g = np.linspace(0.7, 4.5, cplx) * eV

if __name__ == "__main__":

    E_i_1 = np.zeros(cplx)
    E_i_2 = np.zeros(cplx)

    # two different X-factors for two different plots
    X1 = 1
    X2 = 46200
    f = 1/46200
    
    # Run optimization for X=1
    fX1 = f * X1
    print("Running optimization for one sun")
    efficiencies_1 = np.zeros(cplx)
    
    for i in range(cplx):
        eff, temp = Optimize(E_g[i], int_max, T_cell, T_sun, fX1)
        efficiencies_1[i] = -eff  # Convert back to positive
        E_i_1[i] = temp

    # Run optimization for X=46200
    fX2 = f * X2
    print("Running optimization for full concentration")
    efficiencies_2 = np.zeros(cplx)
    
    for i in range(cplx):
        eff, temp = Optimize(E_g[i], int_max, T_cell, T_sun, fX2)
        efficiencies_2[i] = -eff  # Convert back to positive
        E_i_2[i] = temp

    # Create a single figure with both results
    plt.figure(figsize=(12, 8))
    
    # Plot curves
    plt.plot(E_g.m_as('eV'), efficiencies_1 * 100, label='X=1', color='blue')
    plt.plot(E_g.m_as('eV'), efficiencies_2 * 100, label='X=46200', color='red')

    # Get E_g values for peak marking
    E_g_values = E_g.m_as('eV')
    
    max_index1 = np.argmax(efficiencies_1)
    max_eff1 = efficiencies_1[max_index1] * 100
    max_eg1 = E_g_values[max_index1]
    plt.annotate(f'Peak: {max_eff1:.2f}%',
                (max_eg1, max_eff1),
                textcoords="offset points",
                xytext=(0, 20),
                ha='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
    
    # Find and mark peak efficiency for X=4600
    max_index2 = np.argmax(efficiencies_2)
    max_eff2 = efficiencies_2[max_index2] * 100
    max_eg2 = E_g_values[max_index2]
    plt.annotate(f'Peak: {max_eff2:.2f}%',
                (max_eg2, max_eff2),
                textcoords="offset points",
                xytext=(0, 20),
                ha='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    # plotting
    plt.xlabel('Bandgap Energy E_g (eV)')
    plt.ylabel('Efficiency (%)')
    plt.ylim(0, 70)
    plt.title('Efficiency vs Bandgap Energy (E_g) for Different Concentration Factors')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add text annotation explaining the concentration factors
    plt.figtext(0.15, 0.02, 'X=1: One sun concentration\nX=46200: Maximum solar concentration', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.show()
