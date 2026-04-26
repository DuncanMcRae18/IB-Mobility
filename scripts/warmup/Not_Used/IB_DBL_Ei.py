# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import root, differential_evolution
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize, approx_fprime
from pint import UnitRegistry
import numpy as np
import pandas as pd

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

    T_sun = T_sun.to('K').magnitude # Ensure temperature is in Kelvin
    k = u.Quantity('boltzmann_constant').m_as('eV/K') # set the boltzmann constant in eV/K and remove units for calculation

    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0 , cutoff) # Integrate the power input from 0 to a high value

    P_in = fX * (2 * pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4') # Calculate the total power input from the sun

    return P_in

def photon_flux(min, max, T, mu, fX):
    
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    x = np.linspace(min, max, 500)
    y = x**2 / (np.exp((x - mu) / (k * T)) - 1)
    
    integrand = trapezoid(y, x)
    
    N = (fX * 2 * pi / ((h**3) * (c**2))) * integrand * eV**3
    
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

    if flux_val < 0 or flux_conduct < 0:
        return 1e23

    return flux

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):

    """Calculate the current density using the photon flux"""

    conduct_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv * eV, 1)

    if E_i  < E_c:
        inter_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
        inter_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)
    else:
        inter_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1) 
        inter_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        inter_flux = min(inter_flux_C, inter_flux_V)

    flux = conduct_flux + inter_flux
    
    if inter_flux_C < 0 or inter_flux_V < 0 or conduct_flux < 0:
        return 0
    
    J = q * flux
    return J

def find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv):
    def equation(u_ci):
        return IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci)

    guess = np.arange(2.5 * E_i.m_as('eV'), 0.1, -0.5 * E_i.m_as('eV'))  # <-- Added negative step to go downwards

    u_ci = u_cv / 2  # <-- Initialize u_ci

    for g in guess:
        zero = root(equation, x0=g, tol=1e-2)
        if zero.success and 0 < zero.x[0] < min(E_c.m_as('eV'), u_cv):
            u_ci = zero.x[0]
            break
    return u_ci

def efficiency(x, E_i, T_cell, T_sun, int_max, fX, P_in):
    """Calculate dimensionless efficiency"""

    E_g = x[0] * eV
    E_c = E_g - E_i
    u_cv = x[1]
    u_ci = find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv)
    u_iv = u_cv - u_ci

    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    if not 0 < u_cv < E_g.m_as('eV') or not 0 < u_ci < E_c.m_as('eV') or not 0 < u_iv < E_i.m_as('eV'):
        return 0

    if not 0 < efficiency.magnitude < 1:
        return 0

    print(f"E_g={E_g.m_as('eV'):.3f} eV, u_cv={u_cv:.3f} eV, u_ci={u_ci:.3f} eV, u_iv={u_iv:.3f} eV, Efficiency={efficiency.magnitude*100:.2f}%")

    return -efficiency.magnitude  # Negative for minimization

def Optimize(E_i, int_max, T_cell, T_sun, fX):
    P_in = power_input(T_sun, int_max, fX)
    
    # Define bounds for DE (same as before)
    bounds = [(E_i.m_as('eV'), 3.5), (0.5 * E_i.m_as('eV'), 3.5)]
    
    # DE doesn't handle general constraints easily, so enforce in the function
    def constrained_efficiency(x):
        if x[0] <= x[1] or E_i.m_as('eV') < x[1]:  # E_g > u_cv
            return 1  # Penalty for invalid
        return efficiency(x, E_i, T_cell, T_sun, int_max, fX, P_in)
    
    result = differential_evolution(
        constrained_efficiency,
        bounds=bounds,
        maxiter=100,  # Adjust for speed vs. accuracy
        popsize=15,   # Population size (more = better exploration, slower)
        tol=1e-2,     # Tolerance
    )
    
    if result.success:
        return -result.fun, result.x[0]
    else:
        print(f"Optimization failed for E_i={E_i}")
        return 0, 0

def main(E_i, cplx, T_cell, T_sun, int_max, fX, Plot):

    efficiencies = np.zeros(cplx)
    E_g = np.zeros(cplx)

    for i in range(cplx):
        eff, E_g_val = Optimize(E_i[i], int_max, T_cell, T_sun, fX)
        efficiencies[i] = eff
        E_g[i] = E_g_val

    print("\nOptimization complete!")

    if Plot:
        plot_values(E_g, E_i.m_as('eV'), efficiencies * 100)
    return efficiencies

def plot_values(E_g, E_i, efficiencies):
    """Plot efficiency vs intermediate bandgap energy"""
    plt.figure(figsize=(10, 6))
    plt.plot(E_i, efficiencies, marker='o')
    plt.ylim(25, 70)
    plt.xlabel('Intermediate Band Energy E_i (eV)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency of Intermediate Band Solar Cell')
    for x, y, ei in zip(E_i, efficiencies, E_g):
        plt.annotate(f'{ei:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.grid(True)
    plt.show()

"""This portion of this code defines the parameters you wish to use"""

cplx = 9 # complexity of the calculations, higher values increase precision but also computation time

int_max = 10 # designate maximum energy to integrate to

E_i = np.linspace(0.5, 1.3, cplx) * eV

Plot = True # change to true to plot results

fX = f * X # combined solid angle and concentration factor

# E_g = [1.48, 1.71, 1.93, 2.14, 2.36, 2.56, 2.76, 2.96, 3.17] # optimal bandgaps for given intermediate bandgaps from literature
"""E_i = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] # intermediate bandgaps from literature


for i in range(len(E_g)):
    print(-efficiency([E_g[i], E_i[i] * 2.3 ], E_i[i]*eV, T_cell, T_sun, int_max, fX, power_input(T_sun, int_max, fX)))
"""
results = main(E_i, cplx, T_cell, T_sun, int_max, fX, Plot)

