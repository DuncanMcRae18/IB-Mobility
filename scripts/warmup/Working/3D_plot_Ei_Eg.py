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
    return P_in

def photon_flux(min, max, T, mu, fX):
    # simplify constants
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K') # cant use m_as for k, keep getting errors
    mu = mu.m_as('eV')

    "Trapezoidal integration is used for speed"

    # Create array for integration
    x = np.linspace(min, max, 400)

    # integrand
    y = x**2 / (np.exp((x - mu) / (k * T)) - 1)

    # calculate integrand using trapezoid for speed
    integrand = trapezoid(y, x)

    "to use quad integration instead, uncomment below and comment out trapezoid section above"

    # integrand, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max)

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

def efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in):
    #define intermediate band to conduction band gap and chemical potential
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

    return -efficiency.magnitude

def Optimize(E_g, E_i, int_max, T_cell, T_sun, fX):
    # Calculate power input once
    P_in = power_input(T_sun, int_max, fX)

    x = False

    # set bounds for the minimize function
    bounds = [(0.01 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV'))]
    
    # expected value of u_cv
    mean = 0.935 * E_g.m_as('eV')

    # minimize negative efficiency using mean u_cv value
    result = minimize(fun=lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
        x0=[mean],
        method='SLSQP',
        bounds=bounds,
        tol=1e-4)

    # if no good result found, use initial guesses
    if result.fun == 0 or not result.success:

        x = True

        # create initial guesses
        guesses = np.linspace(0.8 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV'), 20)

        # create array for potential u_cv candidates
        candidates = []

        # evaluate efficiency at each guess
        for g in guesses:
            candidates.append((g, efficiency([g], E_g, E_i, T_cell, T_sun, int_max, fX, P_in)))

        # select best candidate
        guess = sorted(candidates, key=lambda x: x[1])[0][0]

        # perform minimization again with new guess
        result = minimize(fun=lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
            x0=[guess],
            method='SLSQP',
            bounds=bounds,
            tol=1e-4)

    # if still no good result found, use differential evolution
    if result.fun == 0 or not result.success:
        bounds = [(0.8 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV'))]
        # Differential evolution parameters optimized for this problem
        result = differential_evolution(
            func=lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
            bounds=bounds,
            strategy='best1bin',
            popsize=30,
            tol=1e-4,
            maxiter=200,
            polish=True
        )
    
    # check if optimization was successful
    if result is not None and result.success and result.fun < 0:
        print(f"Completed optimization for E_g={E_g.m_as('eV'):.2f}, E_i={E_i.m_as('eV'):.2f}, u_cv={result.x[0]:.2f} with max efficiency {-result.fun*100:.8f}%.")
        return -result.fun * 100
    # otherwise return 0 efficiency
    else:
        print(f"Failed optimization for E_g={E_g.m_as('eV'):.2f}, "
              f"E_i={E_i.m_as('eV'):.2f}. Returning 0%.")
        return 0

def optimize_point(args):

    # unpack args
    i, j, E_g, E_i, int_max, T_cell, T_sun, fX = args

    # Check if E_i is less than E_g
    if E_i < 0.99 * E_g:
        # perform optimization
        eff = Optimize(E_g * eV, E_i * eV, int_max, T_cell, T_sun, fX)
        return (i, j, eff)
    return (i, j, 0)

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
        for i, j, eff in results:
            efficiencies[i, j] = eff
            
    return efficiencies

def extrapolate(E_g, E_i, efficiencies):
    
    count = 0

    # loop over all points to find zeros
    for i in range(len(E_i)):
        for j in range(len(E_g)):
            if efficiencies[i, j] == 0 and E_i[i] < 0.99 * E_g[j]:
                # find nearest non-zero neighbors
                count += 1
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(E_i) and 0 <= nj < len(E_g) and efficiencies[ni, nj] != 0:
                            neighbors.append(efficiencies[ni, nj])
                # average neighbors if any found
                if neighbors:
                    efficiencies[i, j] = np.mean(neighbors)

    print(count, " zeros extrapolated out of ", cplx * cplx, " points.")

    return efficiencies

if __name__ == '__main__':
    # concentration factor
    X = 46200
    # solid angle factor
    f = 1/46200
    # combined factor
    fX = f * X

    # maximum energy for integration
    int_max = 10 * eV

    # complexity of the grid
    cplx = 20

    # arrays of conduction band energies and intermediate band energies
    E_g = np.linspace(0.5, 4.5, cplx)
    E_i = np.linspace(0.1, 4.5, cplx)


    # create array for efficiencies
    efficiencies = np.zeros((len(E_g), len(E_i)))

    print("Starting optimization...")

    # for parallelization, each set of values is packed into a 2D array
    args = []
    for i in range(len(E_i)):
        for j in range(len(E_g)):
            args.append((i, j, E_g[j], E_i[i], int_max, T_cell, T_sun, fX))

    # call the parallelization function to begin optimization
    efficiencies = parallelization(args, cplx)

    # extrapolate to fill in zero efficiency points (to make the graph look nice)
    efficiencies = extrapolate(E_g, E_i, efficiencies)

    # print maximum efficiency and corresponding energies
    E_g_max = E_g[np.unravel_index(np.argmax(efficiencies), efficiencies.shape)[1]]
    E_i_max = E_i[np.unravel_index(np.argmax(efficiencies), efficiencies.shape)[0]]
    print(f"Max efficiency: {np.max(efficiencies):.2f}% at E_g = {E_g_max:.2f} eV, E_i = {E_i_max:.2f} eV")

    # create figure
    plt.figure(figsize=(10, 10))
    
    # create meshgrid for plotting
    mesh = plt.pcolormesh(E_g, E_i, efficiencies, shading='auto', cmap='hot')
    
    # Add peak marker
    plt.plot(E_g_max, E_i_max, 'ko', markersize=15, label=f'Peak: {np.max(efficiencies):.2f}%')
    
    # plotting accessories
    plt.colorbar(mesh, label='Efficiency (%)')
    plt.xlabel('E_g (eV)')
    plt.ylabel('E_i (eV)')
    plt.title('Intermediate Band Solar Cell Efficiency Map')
    plt.show()
