# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, minimize, root
from scipy.integrate import quad, trapezoid, IntegrationWarning
from pint import UnitRegistry
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor  # Change import
import time

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
        optimize_point - wrapper function for parallelization to optimize a single (E_g, E_i) point
        parallelization - divides the optimization task across multiple processors
        main - main function to run the calculations and optionally plot and export results"""

def power_input(T_sun, cutoff, fX):
    # remove units from constants
    T_sun = T_sun.to('K').magnitude
    k = u.Quantity('boltzmann_constant').m_as('eV/K')

    # integrate over blackbody spectrum
    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)), 0, cutoff)

    # calculate power input
    P_in = fX * (pi/((h**3)*(c**2))) * integrand * eV ** 4

    # return power
    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    # simplify constants
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    # Create x array for integration
    x = np.linspace(min, max, 400)
    
    # Calculate exponent safely
    with np.errstate(over='ignore', under='ignore'):
        exp_term = (x - mu) / (k * T)
        # Use log sum exp trick for numerical stability
        y = np.where(exp_term < 700,  # threshold for exp overflow
                    x**2 / (np.exp(exp_term) - 1),
                    0.0)  # When exp_term is too large, result is effectively 0

    # Ignore integration warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # calculate integrand using trapezoid rule
        integrand = trapezoid(y, x)

    # calculate photon flux
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
    CB_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv, 1)

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
    n = 20

    # create initial guesses
    guess = np.linspace(minimum + 1e-5, maximum - 1e-5, n)

    # create array for potential zeros
    candidates = []

    # evaluate equation at each guess
    for i in guess:
        ib = equation(i)
        cb = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, i * eV)
        # append both the u_ci value and its current ratio
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
    # if the root is outside of the bounds, return the closest bound since the zero is close to an asymptote
    elif best[0] < (maximum+minimum)/2:
        u_ci = minimum + 1e-5
    elif best[0] > (maximum+minimum)/2:
        u_ci = maximum - 1e-5
    # if no root is found, return false
    else:
        return 0 * eV, False
    return u_ci * eV, True

def efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in):
    #define energies and chemical potentials
    E_c = E_g - E_i
    u_cv = x[0] * eV
    # finding u_ci closest to no IB current
    u_ci, found = find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, int_max)

    # if no valid u_ci found, return 0 efficiency
    if not found:
        return 0
    
    # ensure physical validity of energies
    if E_i <= 0 or E_c <= 0:  
        return 0
    
    # ensure chemical potential is within valid range
    if not 0.01 < u_cv.m_as('eV') < 0.99 * E_g.m_as('eV'):
        return 0

    # calculate output power and efficiency
    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci)
    P_out = J * u_cv / q
    efficiency = P_out / P_in

    # ensure efficiency is within valid range
    if not 0 < efficiency < 1:
        return 0

    return -efficiency.magnitude

def Optimize(E_g, E_i, int_max, T_cell, T_sun, fX):
    # Calculate power input once
    P_in = power_input(T_sun, int_max, fX)

    # set bounds for the minimize function
    bounds = [(0.01, 0.999 * E_g.m_as('eV'))]
    
    # expected value of u_cv
    mean = 0.935 * E_g.m_as('eV')

    # minimize negative efficiency
    result = minimize(fun=lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
        x0=[mean],
        method='SLSQP',
        bounds=bounds,
        tol=1e-3)

    # if no good result found, use initial guesses
    if result.fun == 0 or not result.success:

        # create initial guesses
        guesses = np.linspace(0.8 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV'), 10)

        candidates = []

        for g in guesses:
            candidates.append((g, efficiency([g], E_g, E_i, T_cell, T_sun, int_max, fX, P_in)))

        guess = sorted(candidates, key=lambda x: x[1])[0][0]

        result = minimize(fun=lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
            x0=[guess],
            method='SLSQP',
            bounds=bounds,
            tol=1e-3)

    if result.fun == 0 or not result.success:
        # Differential evolution parameters optimized for this problem
        result = differential_evolution(
            func=lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
            bounds=bounds,
            strategy='best1bin',
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-3,
            maxiter=100,
            polish=True
        )
    
    # check if optimization was successful
    if result is not None and result.success and result.fun < 0:
        print(f"Completed optimization for E_g={E_g.m_as('eV'):.2f}, "
              f"E_i={E_i.m_as('eV'):.2f}, u_cv={result.x[0]:.2f} with max efficiency {-result.fun*100:.8f}%.")
        return -result.fun * 100, result.x[0]
    # otherwise return 0 efficiency
    else:
        print(f"Failed optimization for E_g={E_g.m_as('eV'):.2f}, "
              f"E_i={E_i.m_as('eV'):.2f}. Returning 0%.")
        return 0, 0

def optimize_point(args):
    # unpack args
    i, j, E_g, E_i, int_max, T_cell, T_sun, fX = args

    # Check if E_i is less than E_g
    if E_i < 0.99 * E_g:
        # perform optimization
        eff, u_cv = Optimize(E_g * eV, E_i * eV, int_max, T_cell, T_sun, fX)
        return (i, j, eff, u_cv)
    return (i, j, 0, 0)

def parallelization(args, cplx):
    # Set number of workers (using slightly fewer than CPU cores is good practice)
    n_workers = 10
    
    # Create array for results
    efficiencies = np.zeros((cplx, cplx))
    
    # Create process pool once and use it for all processing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process all args in one go
        results = list(executor.map(optimize_point, args))
        
        # Store results in efficiency array
        for i, j, eff, u_cv in results:
            efficiencies[i, j] = eff
            
    return efficiencies

if __name__ == '__main__':
    # Test parameters
    int_max = 10
    fX = 1

    test_cases = [
        (1.96, 0.34),
        (1.85, 0.34),
        (1.96, 0.28),
        (1.85, 0.28),
        (1.91, 0.28),
        (1.26, 0.28),
        (1.20, 0.28),
        (2.39, 0.22)
    ]

    print("Testing optimization for failing points with debug info...")
    print("-" * 50)

    for E_g_val, E_i_val in test_cases:
        E_g = E_g_val * eV
        E_i = E_i_val * eV
        E_c = E_g - E_i
        
        print(f"\nTesting E_g={E_g_val:.2f}, E_i={E_i_val:.2f}")
        print(f"E_c = {E_c.m_as('eV'):.2f} eV")
        
        # Test u_ci finding at different u_cv values
        test_u_cvs = [0.9, 0.92, 0.94]
        for u_cv_factor in test_u_cvs:
            u_cv = u_cv_factor * E_g
            print(f"\nu_cv = {u_cv.m_as('eV'):.2f} eV ({u_cv_factor:.1f}*E_g)")
            
            u_ci, found = find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, int_max)
            
            if found:
                print(f"Found u_ci = {u_ci.m_as('eV'):.4f} eV")
                # Check current densities
                J_IB = IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci)
                J_CB = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci)
                print(f"J_IB = {J_IB.m_as('A/m^2'):.2e} A/m²")
                print(f"J_CB = {J_CB.m_as('A/m^2'):.2e} A/m²")
            else:
                print("Failed to find valid u_ci")
        
        print("\nRunning full optimization...")
        eff, u_cv = Optimize(E_g, E_i, int_max, T_cell, T_sun, fX)
        print("-" * 50)