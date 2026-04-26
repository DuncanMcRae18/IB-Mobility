# import necessary libraries
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
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

"""This python code is intended to model the detailed balance limit of efficiency for a single junction solar cell
   The functions are as follows:
        photon_flux - calculates the blackbody photon flux given some bandgap E_g, temperature of the material, and the chemical potential of that material
        current_density - calculates the current density using the photon flux
        efficiency - calculates the efficiency for a given bandgap energy and chemical potential
        optimize - optimizes the chemical potential to find the maximum efficiency for each bandgap energy
        export - exports the results to a CSV file
        """

def photon_flux(min, max, T, mu, fX):
    """Calculate photon flux using Gauss-Legendre quadrature"""
    # Convert all inputs to numerical values in proper units
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')
    
    integrand, error = quad(lambda x: (x**2/(np.exp(((x - mu) / (k * T))) - 1)), min, max)
    
    N = (fX * pi / ((h**3) * (c**2))) * integrand * eV**3
    return N # Flux in photons/(m^2*s)

def current_density(E_g, E_C, E_I, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):
    """Calculate the current density using the photon flux"""
    # Direct VB->CB transitions
    flux_in_conduct = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX)
    flux_out_conduct = photon_flux(E_g, int_max * eV, T_cell, u_cv, 1)

    if photon_flux(E_I, E_g, T_sun, 0 * eV, fX) < photon_flux(E_C, E_g, T_sun, 0 * eV, fX):
        flux_in_inter = photon_flux(E_I, E_g, T_sun, 0 * eV, fX)
        flux_out_inter = photon_flux(E_I, E_g, T_cell, u_ci, 1)
    else:
        flux_in_inter = photon_flux(E_C, E_g, T_sun, 0 * eV, fX)
        flux_out_inter = photon_flux(E_C, E_g, T_cell, u_ci, 1)

    flux = (flux_in_conduct + flux_in_inter) - (flux_out_conduct + flux_out_inter)
    J = q * flux
    return J

def power_input(T_sun, cutoff, fX):

    T_sun = T_sun.to('K').magnitude # Ensure temperature is in Kelvin
    k = u.Quantity('boltzmann_constant').m_as('eV/K') # set the boltzmann constant in eV/K and remove units for calculation

    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0 , cutoff) # Integrate the power input from 0 to a high value

    P_in = fX * (pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4') # Calculate the total power input from the sun

    return P_in  # Flux in photons/(m^2·s·m)

def efficiency(u_values, E_g, E_C, E_I, T_cell, T_sun, int_max, fX, P_in):
    """Calculate dimensionless efficiency"""
    
    u_cv = u_values[0]
    u_ci = u_values[1]
    u_iv = u_values[2]  # Add u_iv from the input array
    
    J = current_density(E_g, E_C, E_I, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)  # Pass u_iv
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    return -efficiency.magnitude

def Optimize(E_g, int_max, T_cell, T_sun, fX):
    """Optimize chemical potentials for maximum efficiency"""
        
    P_in = power_input(T_sun, int_max, fX)
    
    # Initial guess
    guess = np.array([
        0.5 * E_g.m_as('eV'),  # u_cv guess
        0.25 * E_g.m_as('eV'),  # u_ci guess
        0.25 * E_g.m_as('eV'),   # u_iv guess
        0.5 * E_g.m_as('eV'),  # E_i guess
        0.25 * E_g.m_as('eV')  # E_c guess
    ])

    
    # Define bounds directly as a list of tuples
    bounds = [
        (0.1 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV')),  # u_cv bounds
        (0.1 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV')),  # u_ci bounds
        (0.1 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV')),  # u_iv bounds
        (0.1 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV')),  # E_i bounds
        (0.1 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV'))   # E_c bounds
    ]


    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - (x[1] + x[2])},  # u_cv = u_ci + u_iv
        {'type': 'ineq', 'fun': lambda x: E_g.m_as('eV') - x[0]},  # u_cv <= E_g
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},  # u_iv <= E_i
        {'type': 'ineq', 'fun': lambda x: x[4] - x[1]}   # u_ci <= E_c
    ]

    def variables(x):
        """Handle optimization variables with proper unit conversion"""
        # Convert optimization variables to quantities with units
        u_cv = x[0] * eV
        u_ci = x[1] * eV
        u_iv = x[2] * eV
        E_C = x[3] * eV
        E_I = x[4] * eV
        
        eff = efficiency([u_cv, u_ci, u_iv], E_g, E_C, E_I, T_cell, T_sun, int_max, fX, P_in)
        return eff if abs(eff) <= 1.0 else 0

    result = minimize(
        variables,
        guess,
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-2, 'maxiter': 1000}  # Reduced iterations
    )

    return -result.fun if result.success else 0


def export(E_g, efficiency): # Export results to a CSV file
    df = pd.DataFrame({'Bandgap Energy (eV)': E_g.magnitude, 'Max Efficiency': efficiency})
    df.to_csv('detailed_balance_limit_data.csv')

"""The last portion of this code defines the parameters you wish to use"""

cutoff = 10 # complexity of the calculations, higher values increase precision but also computation time
E_g = u.Quantity(np.linspace(0.1, 3, cutoff), 'eV') # array of bandgap energies to calculate over
efficiencies = np.zeros(len(E_g)) # create an empty array to store efficiency values
T_sun = 6000 * u.kelvin # temperature of the sun
T_cell = 300 * u.kelvin # temperature of the solar cell
f = 1/(14000*np.pi) # solid angle factor
X = 1 # concentration factor
fX = f * X # combined solid angle and concentration factor
Export = False # change to true to export data to an excel file

"""Main problem solving"""
for i in range(len(E_g.m_as('eV'))):
    
    print('bandgap energy: ',i, 'of' , len(E_g)-1) # print progress to console

    efficiencies[i] = Optimize(E_g[i], cutoff, T_cell, T_sun, fX)


if (Export == True):
    export(E_g, efficiencies)

"""Plotting results"""
plt.figure(figsize=(12, 8))
plt.plot(E_g.magnitude,efficiencies)
plt.xlabel('Bandgap Energy (eV)')
plt.ylabel('Efficiency')
plt.title('Detailed Balance Limit: Efficiency vs Bandgap Energy')
plt.hlines(efficiencies.max(), xmin=0, xmax=3, colors='g', linestyles='dashed', label=f'Max Efficiency: {efficiencies.max():.2%} at {E_g[efficiencies.argmax()].magnitude:.2f} eV')
plt.legend()
plt.grid()
plt.show()
