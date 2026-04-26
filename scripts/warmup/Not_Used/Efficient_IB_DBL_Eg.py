# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import root, differential_evolution
from scipy.integrate import quad, trapezoid, IntegrationWarning
from pint import UnitRegistry
import numpy as np
import pandas as pd
import warnings  # <-- Add this

u = UnitRegistry()

h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt
pi = np.pi

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)

T_sun = 6000 * u.kelvin # temperature of the sun
T_cell = 300 * u.kelvin # temperature of the solar cell
f = 6.8e-5 # solid angle factor


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

    P_in = fX * (pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4') # Calculate the total power input from the sun

    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    """Calculate photon flux using faster integration"""
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    
    x = np.linspace(min, max, 100)  # Reduced from 1000 to 100
    y = x**2 / (np.exp((x - mu) / (k * T)) - 1)
    flux = trapezoid(y, x)

    """
    flux, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)"""

    N = (fX * pi / ((h**3) * (c**2))) * flux * eV**3
    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    u_iv = u_cv - u_ci

    if E_i < E_c:
        flux_val = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci * eV, 1)
    elif E_i == E_c:
        flux_val = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv * eV, 1)
        flux_conduct = 0
    else:
        flux_val = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv * eV, 1)
        flux_conduct = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci * eV, 1)

    flux = (flux_val + flux_conduct).magnitude

    return flux

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):

    """Calculate the current density using the photon flux"""

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

def find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv):
    """Find u_ci by solving IB_current_density = 0 using strict root finding"""
    
    def equation(u_ci):
        return IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci)

    E_c_val = E_c.m_as('eV')
    bound = min(E_c_val, u_cv)

    guesses = np.linspace(0.01 * bound, 0.99 * bound, 10)  # Not multiplied by bound again
    
    for guess in guesses:
        try:
            zero = root(equation, x0=guess, method='hybr', tol=1e-8)
            if zero.success and 0 < zero.x[0] < bound:
                return zero.x[0]
        except Exception:
            continue
    
    return bound * 0.3

def efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in):
    """Calculate dimensionless efficiency"""

    E_i = x[0] * eV
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
    
    return -efficiency.magnitude  # Negative for minimization

def Optimize(E_g, int_max, T_cell, T_sun, fX):
    """Optimization with targeted guesses for specific E_g values"""
    P_in = power_input(T_sun, int_max, fX)
    
    bounds = [(0.01 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV')),  # E_i bounds
              (0.01 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV'))]  # u_cv bounds

    result = differential_evolution(
        lambda x: efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in),
        bounds=bounds,
        maxiter=1000,      
        popsize=30,        
        tol=1e-2,        
        polish=True,    
        init='latinhypercube'
    )


    if result.success:
        return result.fun, result.x[0]
    else:
        print(f"Optimization failed for E_g={E_g_val}")
        return 0, 0

def optimize_wrapper(args):
    # args: (E_g_eV, int_max, T_cell_K, T_sun_K, fX)
    E_g_eV, int_max, T_cell_K, T_sun_K, fX = args
    E_g = E_g_eV * eV
    T_cell = T_cell_K * u.kelvin
    T_sun = T_sun_K * u.kelvin
    return Optimize(E_g, int_max, T_cell, T_sun, fX)

def main(E_g, cplx, T_cell, T_sun, int_max, fX, Plot):
    """Main function with proper parallelization"""
    # Create a list of arguments for each E_g value
    args_list = [
        (E_g[i].m_as('eV'), int_max, T_cell.m_as('K'), T_sun.m_as('K'), fX)
        for i in range(cplx)
    ]

    # Initialize arrays OUTSIDE the loop
    efficiencies = np.zeros(cplx)
    E_i = np.zeros(cplx)

    # Create ThreadPoolExecutor ONCE for all values
    with ThreadPoolExecutor(max_workers=14) as executor:
        results = list(tqdm(executor.map(optimize_wrapper, args_list), 
                            total=len(args_list),
                            desc="Optimizing"))

    # Process all results after parallel execution completes
    for i, (eff, E_i_val) in enumerate(results):
        efficiencies[i] = -eff  # Convert back from negative values used in optimization
        E_i[i] = E_i_val

    print("\nParallel optimization complete!")

    if Plot:
        plot_values(E_g.m_as('eV'), E_i, efficiencies * 100)
        
    return efficiencies

def plot_values(E_g, E_i, efficiencies):
    """Plot E_g vs efficiency and label points with E_i (in eV)."""
    plt.figure(figsize=(10, 6))
    plt.plot(E_g, efficiencies, marker='o', linestyle='-')
    plt.xlabel('Bandgap Energy E_g (eV)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency vs Bandgap Energy (E_g)')
    for x, y, ei in zip(E_g, efficiencies, E_i):
        plt.annotate(f'{ei:.2f} eV', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    plt.grid(True)
    plt.show()


"""This portion of this code defines the parameters you wish to use"""

cplx = 9 # complexity of the calculations, higher values increase precision but also computation time

int_max = 10 # designate maximum energy to integrate to

E_g = np.linspace(0.7, 4.5, cplx) * eV  # Focused on high-efficiency range

Plot = True # change to true to plot results

if __name__ == '__main__':
    # Only run for X=1
    X = 1
    fX_val = f * X
    
    # Run the optimization with plotting enabled
    result = main(E_g, cplx, T_cell, T_sun, int_max, fX_val, True)
    
    # If you want to customize the plot after running the optimization
    plt.figure(figsize=(10, 6))
    plt.plot(E_g.m_as('eV'), result * 100, marker='o', linestyle='-', label='X=1')
    plt.xlabel('Bandgap Energy E_g (eV)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency vs Bandgap Energy (E_g) for X=1')
    plt.grid(True)
    plt.legend()
    plt.show()
