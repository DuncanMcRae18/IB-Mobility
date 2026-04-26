# import necessary libraries
from scipy.optimize import root
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np

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