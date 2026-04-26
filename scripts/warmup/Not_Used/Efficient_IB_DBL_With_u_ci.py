# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, minimize, root
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np
import pandas as pd
import warnings
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

    # remove units from constants
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    # Integrate over blackbody spectrum
    flux, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)

    # determine flux
    N = (fX * pi / ((h**3) * (c**2))) * flux * eV**3

    # return flux
    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    # Calculate u_iv from u_cv and u_ci
    u_iv = u_cv - u_ci

    # if statememnt to choose limits of integration
    if E_i < E_c:
        flux_val = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
    else:
        flux_val = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1)

    # determine total flux
    flux = (flux_val + flux_conduct).magnitude

    # return flux
    return flux

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):

    # conduction flux is simple to calculate
    conduct_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv * eV, 1)

    # if statement to choose limits of integration, then choose minimum flux to ensure current balance
    if E_i  <= E_c:
        inter_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
        inter_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)

    else:
        inter_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1) 
        inter_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)

    # calculate total flux
    flux = conduct_flux + inter_flux

    # calculate current density
    J = q * flux

    # return current density
    return J

def find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv):

    def equation(u_ci):
        return IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci)

    guess = np.arange(min(E_c.m_as('eV'), u_cv), 0.1, -0.2 * E_i.m_as('eV')) 

    for g in guess:
        zero = root(equation, x0=g, tol=1e-10)
        if zero.success and 0 < zero.x[0] < min(E_c.m_as('eV'), u_cv):
            u_ci = zero.x[0]
            break
    return 0

def efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in):

    # simopify constants and variables
    E_i = x[0] * eV
    E_c = E_g - E_i
    u_cv = x[1]
    u_ci = x[2]
    u_iv = u_cv - u_ci

    # check to ensure band gaps are valid
    if E_i <= 0 or E_c <= 0:  
        return 0
    
    # check to ensure potentials are valid
    if not (0 < u_cv < E_g.m_as('eV')) or not (0 < u_ci < min(E_c.m_as('eV'),u_cv)) or not (0 < u_iv < min(E_i.m_as('eV'),u_cv)):  # Valid conduction band potential
        return 0

    # calculate output power and efficiency
    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    # ensure that efficiency is within physical bounds
    if not 0 < efficiency.magnitude < 1:  # Theoretical max ~68% for ideal solar cells
        return 0
    
    # return negative efficiency
    return -efficiency.magnitude  # Negative for minimization

def Optimize(E_g, int_max, T_cell, T_sun, fX):
    # Calculate power input once
    P_in = power_input(T_sun, int_max, fX)
    
    # simplify E_g without units 
    E_g_val = E_g.m_as('eV')

    # set bounds for E_i, u_cv, and u_ci
    bounds = [
        (0.01 * E_g_val, 0.99 * E_g_val),   # E_i bounds
        (0.01 * E_g_val, 0.99 * E_g_val),   # u_cv bounds
        (0.01 * E_g_val, 0.99 * E_g_val)    # u_ci bounds
    ]
    
    #minimization using differential evolution
    guess = differential_evolution(
        lambda x: efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in),
        bounds=bounds,
        strategy='best1bin',
        maxiter=700,
        popsize=10,
        tol= 1e-2,
        polish=False,
    )

    result = minimize(efficiency,
                          guess.x,
                          args=(E_g, T_cell, T_sun, int_max, fX, P_in),
                          bounds=bounds, tol=1e-3, method='SLSQP')

    if result.success:
        print(f"Optimization successful for E_g={E_g.m_as('eV')}: Efficiency={-result.fun*100:.2f}%, E_i={result.x[0]:.2f} eV, u_cv={result.x[1]:.2f} eV, u_ci={result.x[2]:.2f} eV")
        return result.fun, result.x[0]
    else:
        print(f"Optimization failed for E_g={E_g.m_as('eV')}")
        return 0, 0


"""This portion of this code defines the parameters you wish to use"""

cplx = 20 # complexity of the calculations, higher values increase precision but also computation time

int_max = 10 # designate maximum energy to integrate to

E_g = np.linspace(0.7, 4.5, cplx) * eV  # Focused on high-efficiency range

if __name__ == '__main__':

    E_i_1 = np.zeros(cplx)
    E_i_2 = np.zeros(cplx)

    # two different X-factors for two different plots
    X1 = 1
    X2 = 46200
    f = 6.7e-5
    
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
    
    # Find and mark peak efficiency for X=1
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