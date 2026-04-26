# import necessary libraries
from unittest import result
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from pint import UnitRegistry
import numpy as np
import pandas as pd

# Simplify unit registry name
u = UnitRegistry()

h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt
pi = np.pi
T_sun = 6000 * u.kelvin # temperature of the sun
T_cell = 300 * u.kelvin # temperature of the solar cell
f = 1/(14000*np.pi) # solid angle factor
"""This python code is intended to model the detailed balance limit of efficiency for an intermediate band solar cell
   The functions are as follows:
        photon_flux - calculates the blackbody photon flux given some bandgap E_g, temperature of the material, and the chemical potential of that material
        current_density - calculates the current density using the photon flux
        power_input - calculates the total power input from the sun
        efficiency - calculates the efficiency for a given bandgap energy and chemical potential
        optimize - optimizes the chemical potential to find the maximum efficiency for each bandgap energy
        main - main function to run the calculations and optionally plot and export results
        plot_values - plots the efficiency vs intermediate bandgap energy
        export - exports the results to a CSV file"""

def photon_flux(min, max, T, mu, fX):
    """Calculate photon flux using Gauss-Legendre quadrature"""
    
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')
    epsilon = 1e-10  # Small value to prevent division by zero
    
    integrand, error = quad(lambda x: (x**2/(np.exp(((x - mu) / (k * T))) - 1 + epsilon)), min, max)

    N = (fX * pi / ((h**3) * (c**2))) * integrand * eV**3

    return N # Flux in photons/(m^2*s)

def current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):
    """Calculate the current density using the photon flux"""

    conduct_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv * eV, 1)

    if E_i  < E_c:
        flux_in_A = photon_flux(E_i, E_c, T_sun, 0 * eV, fX)
        flux_out_A = photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        flux_in_B = photon_flux(E_c, E_g, T_sun, 0 * eV, fX)
        flux_out_B = photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
        if flux_in_A - flux_out_A < flux_in_B - flux_out_B:
            inter_flux = flux_in_A - flux_out_A
        else:
            inter_flux = flux_in_B - flux_out_B
    else:
        flux_in_A = photon_flux(E_c, E_i, T_sun, 0 * eV, fX)
        flux_out_A = photon_flux(E_c, E_i, T_cell, u_ci * eV, 1)
        flux_in_B = photon_flux(E_i, E_g, T_sun, 0 * eV, fX)
        flux_out_B = photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        if photon_flux(E_c, E_i, T_sun, 0 * eV, fX) < photon_flux(E_i, E_g, T_sun, 0 * eV, fX):
            inter_flux = flux_in_A - flux_out_A
        else:
            inter_flux = flux_in_B - flux_out_B

    flux = conduct_flux + inter_flux
    J = q * flux
    return J

def power_input(T_sun, cutoff, fX):

    T_sun = T_sun.to('K').magnitude # Ensure temperature is in Kelvin
    k = u.Quantity('boltzmann_constant').m_as('eV/K') # set the boltzmann constant in eV/K and remove units for calculation

    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0 , cutoff) # Integrate the power input from 0 to a high value

    P_in = fX * (pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4') # Calculate the total power input from the sun

    return P_in  # Flux in photons/(m^2·s·m)

def efficiency(u_values, E_g, E_c, E_i, T_cell, T_sun, int_max, fX, P_in):
    """Calculate dimensionless efficiency"""
    
    u_cv = u_values[0]
    u_ci = u_values[1]
    u_iv = u_cv - u_ci

    J = current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    return -efficiency.magnitude

def Optimize(E_g, E_c, E_i,int_max, T_cell, T_sun, fX):
    """Optimize chemical potentials for maximum efficiency"""
        
    P_in = power_input(T_sun, int_max, fX)
    
    # Initial guess
    guess = np.array([
        0.5 * E_g.m_as('eV'),  # u_cv guess
        0.5 * E_i.m_as('eV')   # u_ci guess
    ])

    
    # Define bounds directly as a list of tuples
    bounds = [
        (0.1 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV')),  # u_cv bounds
        (0.1 * E_i.m_as('eV'), 0.9 * E_i.m_as('eV'))   # u_ci bounds
    ]

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # u_cv >= u_ci
        {'type': 'ineq', 'fun': lambda x: E_g.m_as('eV') - x[0]},  # u_cv <= E_g
        {'type': 'ineq', 'fun': lambda x: E_i.m_as('eV') - x[1]}]   # u_ci <= E_i

    def variables(x):
        eff = efficiency([x[0], x[1]], E_g, E_c, E_i, T_cell, T_sun, int_max, fX, P_in)
        return eff if abs(eff) <= 1.0 else 0

    result = minimize(
        variables,
        guess,
        method='SLSQP',
        bounds=bounds,
        options={'gtol': 1e-3, 'maxiter': 100}  # Reduced iterations
    )

    return -result.fun if result.success else 0

def main(E_g, E_i, cplx, T_cell, T_sun, int_max, fX, Plot):
    """Main problem solving with proper energy handling"""
    
    efficiencies = np.zeros((cplx, cplx))
    total = cplx * cplx
    
    for i in range(cplx):
        for j in range(cplx):
            print(f"Progress: {i*cplx + j + 1}/{total}", end='\r')
            # Set efficiency to 0 for physically impossible cases
            if E_i[j] >= E_g[i]:
                efficiencies[i, j] = 0
            else:
                efficiencies[i, j] = Optimize(E_g[i], E_i[j], E_g[i] - E_i[j], int_max, T_cell, T_sun, fX)

    print("\nOptimization complete!")
    
    print('Efficiencies (%):')
    for i in range(cplx):
        for j in range(cplx):
            print(f"E_g: {E_g[i].m_as('eV')}, E_i: {E_i[j].m_as('eV')}, Efficiency: {efficiencies[i, j]:.2f}%")

    if Plot:
        plot_values(E_g.m_as('eV'), efficiencies)

    return efficiencies

def plot_values(E_g, efficiencies):
    """Plot efficiency vs intermediate bandgap energy"""
    plt.figure(figsize=(10, 6))
    # Swap E_g range in extent parameter to correctly map axes
    plt.imshow(efficiencies.T, extent=(E_g[0], E_g[-1], E_g[0], E_g[-1]), 
              origin='lower', aspect='auto', cmap='hot', vmin=0, vmax=efficiencies.max())
    plt.colorbar(label='Efficiency (%)')
    plt.xlabel('Intermediate Band Energy E_i (eV)')
    plt.ylabel('Bandgap Energy E_g (eV)')
    plt.title('Efficiency of Intermediate Band Solar Cell')
    plt.grid(False)
    plt.show()

"""This portion of this code defines the parameters you wish to use"""

cplx = 10 # complexity of the calculations, higher values increase precision but also computation time

int_max = 10 # designate maximum energy to integrate to

E_g = np.linspace(0.6, 4, cplx) * eV
E_i = np.linspace(0, 4, cplx) * eV

X = 1 # concentration factor

Plot = True # change to true to plot results

fX = f * X # combined solid angle and concentration factor

results = main(E_g, E_i, cplx, T_cell, T_sun, int_max, fX, Plot)


