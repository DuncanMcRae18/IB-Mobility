# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, minimize, root
from scipy.integrate import quad, trapezoid
from pint import UnitRegistry
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor

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
        rough_efficiency - calculates a rough efficiency using a free u_ci
        efficiency - calculates the efficiency for a given bandgap energy and chemical potential
        optimize - optimizes the chemical potential to find the maximum efficiency for each bandgap energy
        optimize_point - wrapper function for parallelization to call (has to be outside the function)
        parallelization - divides the optimization task across multiple processors for speed
        extrapolate - fills in zero efficiency points based on neighboring values
        main - main function to run the calculations and optionally plot and export results"""

def power_input(T_sun, cutoff, fX):
    # remove units from constants
    T_sun = T_sun.to('K').magnitude
    k = u.Quantity('boltzmann_constant').m_as('eV/K')
    cutoff = cutoff.m_as('eV')

    # integrate over blackbody spectrum
    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0, cutoff)

    # calculate power input
    P_in = fX * (pi/((h**3)*(c**2))) * integrand * eV ** 4

    # return power
    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    # remove units from values
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    # calculate integrand for the photon flux
    integrand, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)

    # calculate the total flux
    N = (fX * pi / ((h**3) * (c**2))) * integrand * eV**3

    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    # define valance to intermediate chemical potential
    u_iv = u_cv - u_ci

    # calculate current densities in each transition based on E_i position
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

    # calculate net flux
    flux = (flux_conduct_abs - flux_conduct_emit) + (flux_val_emit - flux_val_abs)

    # calculate current density
    J = q * flux

    return J

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci):

    # define valance to intermediate chemical potential
    u_iv = u_cv - u_ci

    # calculate direct conduction band flux
    CB_flux = photon_flux(E_g, int_max, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max, T_cell, u_cv, 1)

    # calculate intermediate band flux
    if E_i  <= E_c:
        IB_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci, 1)
        IB_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv, 1)

    else:
        IB_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci, 1) 
        IB_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv, 1)
    
    # use minimum flux to ensure consrevation of carriers
    IB_flux = min(IB_flux_C, IB_flux_V)

    # total flux
    flux = CB_flux + IB_flux

    # calculate current density
    J = q * flux

    return J

def find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, int_max):
    # define equation for IB current
    def equation(u_ci):
        return IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci * eV)

    # bounds based on the limits of u_ci and u_iv
    maximum = min(E_c, u_cv).m_as('eV')
    minimum = u_cv.m_as('eV') - min(E_i, u_cv).m_as('eV')
    
    # number of initial guesses
    n = 10

    # create guesses array
    guess = np.linspace(minimum + 1e-5, maximum - 1e-5, n)

    # create array for potential zeros
    candidates = []

    # evaluate equation at each guess
    for i in guess:
        ib = equation(i)
        cb = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, i * eV)
        # append both the u_ci value and its ratio
        candidates.append((i, (ib / cb)))

    # if candidates is empty, return false
    if not candidates:
        return minimum * eV, False

    # find best candidate closest to zero
    best = min(candidates, key=lambda x: abs(x[1]))

    # find root using best candidate
    zero = root(equation, best[0], method='hybr')
    # if the root finder succeeds and is within bounds, return it
    if zero.success and minimum < zero.x[0] < maximum:
        u_ci = zero.x[0]
    
    # if the root isn't found within the bounds, return 0 efficiency
    else:
        return 0 * eV, False
    return u_ci * eV, True

def rough_efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in):

    # simplify constants and variables
    E_i = x[0] * eV
    E_c = E_g - E_i
    u_cv = x[1] * eV
    u_ci = x[2] * eV
    u_iv = u_cv - u_ci

    tol = 0.01

    # check to ensure band gaps are valid
    if E_i <= tol * eV or E_c <= tol * eV:  
        return 0

    # check to ensure potentials are valid
    if not (tol * eV < u_cv < (1-tol) * E_g) or not (tol * eV < u_ci < (1-tol) * min(E_c,u_cv)) or not (tol * eV < u_iv < (1-tol) * min(E_i,u_cv)): 
        return 0

    # calculate output power and efficiency
    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci)
    P_out = J * u_cv * eV / q
    efficiency = P_out / P_in

    # ensure that efficiency is practical
    if not 0 < efficiency.magnitude < 1:
        return 0
    
    # return negative efficiency for minimization
    return -efficiency.magnitude

def efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in):
    # simplify constants and variables
    E_i = x[0] * eV
    E_c = E_g - E_i
    u_cv = x[1] * eV
    u_ci, valid = find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, int_max) 

    # if no u_ci is found, efficiency is 0
    if not valid:
        return 0

    tol = 0.01

    # check to ensure band gaps are valid
    if E_i <= tol * eV or E_c <= tol * eV:  
        return 0

    # check to ensure potentials are valid
    if not (tol * eV < u_cv < (1-tol) * E_g):
        return 0

    # calculate output power and efficiency
    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci)
    P_out = J * u_cv / q
    efficiency = P_out / P_in

    # ensure efficiency is practical
    if not 0 < efficiency < 1:
        return 0

    # return negative efficiency for minimization
    return -efficiency

def Optimize(E_g, int_max, T_cell, T_sun, fX):
    # Calculate power input once per optimizaition
    P_in = power_input(T_sun, int_max, fX)

    # simplify E_g without units
    E_g_val = E_g.m_as('eV')

    # Bounds for approximating E_i and u_cv
    rough_bounds = [
        (0.01, 0.99 * E_g_val),   # E_i bounds
        (0.01, 0.99 * E_g_val),   # u_cv bounds
        (0.01, 0.99 * E_g_val),   # u_ci bounds
    ]

    # find a rough guess
    guess = differential_evolution(
        lambda x: rough_efficiency(x, E_g, T_cell, T_sun, int_max, fX, P_in),
        bounds=rough_bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=30,
        tol=1e-2,
        polish=False)

    # bounds for finding exact values
    bounds = [
        (0.01, 0.99 * E_g_val),   # E_i bounds
        (0.01, 0.99 * E_g_val),   # u_cv bounds
    ]

    # find the exact peak value
    result = minimize(efficiency, guess.x[:2],
                args=(E_g, T_cell, T_sun, int_max, fX, P_in),
                bounds=bounds, tol=1e-8,
                method='SLSQP')

    # output results if successful
    if result.success and result.fun < 0:
        print(f"Optimization successful for E_g={E_g_val}: Efficiency={-result.fun*100:.8f}%, E_i={result.x[0]:.8f} eV, u_cv={result.x[1]:.8f} eV")
        return result.fun

    # fallback to rough guess if optimization fails
    elif guess.success and guess.fun < 0:
        print(f"Optimization fell back to rough guess for E_g={E_g.m_as('eV')}: Efficiency={-guess.fun*100:.8f}%, E_i={guess.x[0]:.8f} eV, u_cv={guess.x[1]:.8f} eV")
        return guess.fun
    
    # otherwise the optimization fails
    else:
        print(f"Optimization failed for E_g={E_g.m_as('eV')}")
        return 0

if __name__ == "__main__":

    """This portion of this code defines the parameters you wish to use"""
    cplx = 50
    int_max = 100 * eV
    E_g = u.Quantity(np.linspace(0.95, 4.5, cplx), 'eV')

    # two different X-factors for two different plots
    X1 = 1
    X2 = 46200
    f = 1/46200
    
    # Run optimization for X=1
    fX1 = f * X1
    print("Running optimization for one sun")
    efficiencies_1 = np.zeros(cplx)
    
    for i in range(cplx):
        eff = Optimize(E_g[i], int_max, T_cell, T_sun, fX1)
        efficiencies_1[i] = -eff  # Convert back to positive

    # Run optimization for X=46200
    fX2 = f * X2
    print("Running optimization for full concentration")
    efficiencies_2 = np.zeros(cplx)
    
    for i in range(cplx):
        eff = Optimize(E_g[i], int_max, T_cell, T_sun, fX2)
        efficiencies_2[i] = -eff  # Convert back to positive

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
    plt.annotate(f'Peak: {max_eff1:.4f}%',
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
    plt.annotate(f'Peak: {max_eff2:.4f}%',
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
    plt.figtext(0.15, 0.02, 'X=1: One sun concentration\nX=46200: Maximum solar concentration', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    plt.show()