# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor

u = UnitRegistry()

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
    T_sun = T_sun.to('K').magnitude
    k = u.Quantity('boltzmann_constant').m_as('eV/K')

    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0, cutoff)

    # ADDED FACTOR OF 2 for polarization states of light
    P_in = fX * (2 * pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4')

    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    """Calculate photon flux using faster integration"""
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    flux, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)

    # ADDED FACTOR OF 2 for polarization states of light
    N = (fX * 2 * pi / ((h**3) * (c**2))) * flux * eV**3
    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    u_iv = u_cv - u_ci

    if E_i < E_c:
        flux_val = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
    else:
        flux_val = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1)

    flux = (flux_val + flux_conduct).magnitude

    return flux

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):

    conduct_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv * eV, 1)

    if E_i  <= E_c:
        inter_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
        inter_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)

    else:
        inter_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1) 
        inter_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)

    flux = conduct_flux + inter_flux

    J = q * flux
    return J

def efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in):
    """Calculate dimensionless efficiency with u_ci as an optimization variable"""
    E_i = x[0] * eV
    E_c = E_g - E_i
    u_cv = x[1]
    u_ci = x[2]  # Now directly from optimization vector
    u_iv = u_cv - u_ci

    # Basic validation of parameters
    if E_i <= 0 or E_c <= 0:  # Valid bandgap partitioning
        return 0

    if not (0 < u_cv < E_g.m_as('eV')) or not (0 < u_ci < min(E_c.m_as('eV'),u_cv)) or not (0 < u_iv < min(E_i.m_as('eV'),u_cv)):  # Valid conduction band potential
        return 0
    
    # Check intermediate band current balance
    ib_current = IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci)
    if abs(ib_current) > 1e-1:  # Allow some tolerance but penalize large imbalance
        return 0

    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    if not 0 < efficiency.magnitude < 0.99:  # Theoretical max ~68% for ideal solar cells
        return 0
    
    return -efficiency.magnitude  # Negative for minimization

def Optimize(E_g, int_max, T_cell, T_sun, fX):
    """Optimization with u_ci as a direct variable"""
    P_in = power_input(T_sun, int_max, fX)
    
    E_g_val = E_g.m_as('eV')

    bounds = [
        (0.01 * E_g_val, 0.99 * E_g_val),   # E_i bounds
        (0.01 * E_g_val, 0.99 * E_g_val),   # u_cv bounds
        (0.01 * E_g_val, 0.99 * E_g_val)    # u_ci bounds
    ]

    result = minimize(efficiency,
    args=(E_g, T_cell, T_sun, int_max, fX, P_in),
    bounds=bounds,
    method='L-BFGS-B',
    options={'maxiter': 100, 'disp': False})
    
    if result.success:
        print(f"Optimization successful for E_g={E_g.m_as('eV')}: Efficiency={-result.fun*100:.2f}%, E_i={result.x[0]:.2f} eV, u_cv={result.x[1]:.2f} eV, u_ci={result.x[2]:.2f} eV")
        return result.fun, result.x[0]
    else:
        print(f"Optimization failed for E_g={E_g.m_as('eV')}")
        return 0, 0

def optimize_wrapper(args):
    # args: (E_g_eV, int_max, T_cell_K, T_sun_K, fX)
    E_g_eV, int_max, T_cell_K, T_sun_K, fX = args
    E_g = E_g_eV * eV
    T_cell = T_cell_K * u.kelvin
    T_sun = T_sun_K * u.kelvin
    return Optimize(E_g, int_max, T_cell, T_sun, fX)

# Removed redundant main() and plot_values() functions
# All functionality is handled directly in the __main__ block below


"""This portion of this code defines the parameters you wish to use"""

cplx = 10 # complexity of the calculations, higher values increase precision but also computation time

int_max = 10 # designate maximum energy to integrate to

E_g = np.linspace(0.7, 4.5, cplx) * eV  # Focused on high-efficiency range

if __name__ == '__main__':
    # Track efficiency and E_i values
    E_i_values1 = np.zeros(cplx)
    E_i_values2 = np.zeros(cplx)
    
    # Correct concentration factors
    X1 = 1
    X2 = 4600
    f = 1/(2*pi*14000)  # Base solid angle factor
    
    # Run optimization for X=1
    fX_val1 = f * X1
    print("Running optimization for X=1...")
    
    # Run main function with modified structure to track E_i values
    args_list = [(E_g[i].m_as('eV'), int_max, T_cell.m_as('K'), T_sun.m_as('K'), fX_val1)
                for i in range(cplx)]
    
    result1 = np.zeros(cplx)
    with ThreadPoolExecutor(max_workers=14) as executor:
        results = list(executor.map(optimize_wrapper, args_list))
    
    for i, (eff, e_i) in enumerate(results):
        result1[i] = -eff  # Convert back from negative
        E_i_values1[i] = e_i
        
    # Run optimization for X=4600
    fX_val2 = f * X2  # Correctly scaled by X2=4600
    print("Running optimization for X=4600...")
    
    args_list = [(E_g[i].m_as('eV'), int_max, T_cell.m_as('K'), T_sun.m_as('K'), fX_val2)
                for i in range(cplx)]
    
    result2 = np.zeros(cplx)
    with ThreadPoolExecutor(max_workers=14) as executor:
        results = list(executor.map(optimize_wrapper, args_list))
    
    for i, (eff, e_i) in enumerate(results):
        result2[i] = -eff  # Convert back from negative
        E_i_values2[i] = e_i
    
    # Create a single figure with both results
    plt.figure(figsize=(12, 8))
    
    # Plot curves
    plt.plot(E_g.m_as('eV'), result1 * 100, marker='o', linestyle='-', label='X=1', color='blue')
    plt.plot(E_g.m_as('eV'), result2 * 100, marker='s', linestyle='--', label='X=4600', color='red')
    
    # Get E_g values for peak marking
    E_g_values = E_g.m_as('eV')
    
    # Find and mark peak efficiency for X=1
    max_index1 = np.argmax(result1)
    max_eff1 = result1[max_index1] * 100
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
    max_index2 = np.argmax(result2)
    max_eff2 = result2[max_index2] * 100
    max_eg2 = E_g_values[max_index2]
    plt.annotate(f'Peak: {max_eff2:.2f}%',
                (max_eg2, max_eff2),
                textcoords="offset points",
                xytext=(0, 20),
                ha='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Standard plot elements
    plt.xlabel('Bandgap Energy E_g (eV)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency vs Bandgap Energy (E_g) for Different Concentration Factors')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add text annotation explaining the concentration factors
    plt.figtext(0.15, 0.02, 'X=1: One sun concentration (AM1.5G)\nX=4600: Maximum solar concentration', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.show()