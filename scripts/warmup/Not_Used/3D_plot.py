# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, minimize, root
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor

# simplify unit registry
u = UnitRegistry()

# simplify constants
h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt
pi = np.pi

global uci

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
    P_in = fX * (pi/((h**3)*(c**2))) * integrand * eV ** 4

    # return power
    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux, error = quad(lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), min, max, limit=100)

    N = (fX * pi / ((h**3) * (c**2))) * flux * eV**3

    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    u_iv = u_cv - u_ci

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


    flux = (flux_conduct_abs - flux_conduct_emit) + (flux_val_emit - flux_val_abs)

    J = q * flux

    return J

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci):

    u_iv = u_cv - u_ci

    CB_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv, 1)

    if E_i  <= E_c:
        IB_flux_C = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci, 1)
        IB_flux_V = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv, 1)

    else:
        IB_flux_C = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci, 1) 
        IB_flux_V = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv, 1)
    
    IB_flux = min(IB_flux_C, IB_flux_V)

    flux = CB_flux + IB_flux

    J = q * flux

    # return current density
    return J

def find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv):
    def equation(u_ci):
            return IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci * eV)

    maximum = min(E_c, u_cv).m_as('eV')
    minimum = u_cv.m_as('eV') - min(E_i, u_cv).m_as('eV')
    n = 20

    guess = np.linspace(minimum + 1e-5, maximum - 1e-5, n)
    candidates = []

    for i in guess:
        if u_cv - (i * eV) < E_i and i < E_c.m_as('eV'):
            ib = abs(equation(i).m_as('A/m^2'))
            cb = abs(CB_current_density(E_g, E_c, E_i, 10, T_cell, T_sun, fX, u_cv, i * eV).m_as('A/m^2'))
            candidates.append((i, abs(ib / cb)))

    candidates = sorted(candidates, key=lambda x: x[1])[:3]

    roots = []
    for i in candidates:
        zero = root(equation, i[0], method='hybr', tol=1e-8)
        if not zero.success:
            zero = root(equation, i[0], method='lm', tol=1e-8)
        if zero.success:
            val = zero.x[0]
            cb = CB_current_density(E_g, E_c, E_i, 10, T_cell, T_sun, fX, u_cv, val * eV).m_as('A/m^2')
            ib = equation(val).m_as('A/m^2')

            if (cb > 0 and ib < 1e-10):
                roots.append((val, ib/cb))

    if len(roots) == 0:
        if len(candidates) > 0:
            for cand, curr in candidates:
                if curr < 0:
                    print("Found valid candidate u_ci value:", cand, "with current ratio =", curr)
                    return cand * eV, True
        return 0 * eV, False

    u_ci = sorted(roots, key=lambda x: abs(x[0]))[0]

    if u_ci[0] == minimum or u_ci[0] == maximum:
        print("u_ci found at boundary:", u_ci[0])

    return u_ci[0] * eV, True

def rough_efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in):
    # simopify constants and variables
    E_c = E_g - E_i
    u_cv = x[0] * eV
    u_ci = x[1] * eV
    u_iv = u_cv - u_ci

    # check to ensure band gaps are valid
    if E_i <= 0 or E_c <= 0:  
        return 0

    # check to ensure potentials are valid
    if not (0 * eV < u_cv < E_g) or not (0 * eV < u_ci < min(E_c,u_cv)) or not (0 * eV < u_iv < min(E_i,u_cv)): 
        return 0

    # calculate output power and efficiency
    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci)
    P_out = J * u_cv / q
    efficiency = P_out / P_in

    # ensure that efficiency is within physical bounds
    if not 0 < efficiency.magnitude < 1:  # Theoretical max ~68% for ideal solar cells
        return 0
    
    # return negative efficiency
    return -efficiency.magnitude

def efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in):
    E_c = E_g - E_i
    u_cv = x[0] * eV
    u_ci, found = find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv) 

    if not found:
        return 0

    if E_i <= 0 or E_c <= 0:  
        return 0

    if not (0.1 < u_cv.m_as('eV') < 0.9 * E_g.m_as('eV')):
        return 0

    J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci)
    P_out = J * u_cv / q
    efficiency = P_out / P_in

    if not 0 < efficiency < 1:
        return 0

    return -efficiency.magnitude

def Optimize(E_g, E_i, int_max, T_cell, T_sun, fX):
    # Calculate power input once
    P_in = power_input(T_sun, int_max, fX)
    E_c = E_g - E_i

    rough_bounds = [(0.01 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV')),
                    (0.01 * E_c.m_as('eV'), 0.99 * E_c.m_as('eV'))]

    n = 5

    E_i_est = np.linspace(0.3 * E_g.m_as('eV'), 0.5 * E_g.m_as('eV'), 5)
    u_cv_est = np.linspace(0.5 * E_g.m_as('eV'), 0.9 * E_g.m_as('eV'), 5)
    u_ci_est = np.linspace(0.4 * E_g.m_as('eV'), 0.6 * E_g.m_as('eV'), 5)

    initial = np.column_stack([E_i_est, u_cv_est, u_ci_est])


    rough_guess = differential_evolution(
        lambda x: rough_efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
        bounds=rough_bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=20,
        tol=1e-1,
        polish=False,
        init=initial
    )

    print(f"Rough guess for E_g={E_g.m_as('eV')}: u_cv={rough_guess.x[0]:.8f} eV, u_ci={rough_guess.x[1]:.8f} eV, Efficiency={-rough_guess.fun*100:.8f}%")

    bounds = [(0.01 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV'))]

    guess = differential_evolution(
        lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
        bounds=bounds,
        x0 = [rough_guess.x[0]],
        maxiter=50,
        popsize=10,
        tol=1e-1)

    if guess.success and guess.fun < 0:
        print(f"Rough optimization successful for E_g={E_g.m_as('eV')}: Efficiency={-guess.fun*100:.8f}%, E_i={E_i.m_as('eV'):.8f} eV, u_cv={guess.x[0]:.8f} eV")
    else:
        print(f"Rough optimization failed for E_g={E_g.m_as('eV')}")
        guess.x = [0.5 * E_g.m_as('eV')]

    result = minimize(
        lambda x: efficiency(x, E_g, E_i, T_cell, T_sun, int_max, fX, P_in),
        x0=[guess.x[0]],
        bounds=bounds,
        method='SLSQP',
        tol=1e-2)

    if result.success and result.fun <= guess.fun:
        print(f"\033[92mOptimization successful for E_g={E_g.m_as('eV')}: Efficiency={-result.fun*100:.8f}%, E_i={E_i.m_as('eV'):.8f} eV, u_cv={result.x[0]:.8f} eV\033[0m")
        return result.fun
    elif guess.success:
        print(f"\033[93mOptimization failed for E_g={E_g.m_as('eV')}, using guess, Efficiency={-guess.fun*100:.8f}.\033[0m")
        return guess.fun
    else:
        print(f"\033[91mOptimization failed for E_g={E_g.m_as('eV')}, using rough guess, Efficiency={-rough_guess.fun*100:.8f}.\033[0m")
        return rough_guess.fun

def optimize_point(args):
    i, j, E_g_val, E_i_val, int_max, T_cell, T_sun, fX = args
    if E_i_val < E_g_val:
        print(f"Optimizing for E_g={E_g_val:.2f} eV, E_i={E_i_val:.2f} eV")
        result = Optimize(E_g_val * eV, E_i_val * eV, int_max, T_cell, T_sun, fX)
        return (i, j, -result * 100)
    else:
        return (i, j, 0)

if __name__ == '__main__':
    X = 1
    f = 1

    fX = f * X

    int_max = 10

    cplx = 20

    E_g = np.linspace(0.1, 4.5, cplx)
    E_i = np.linspace(0.1, 4.5, cplx)

    efficiencies = np.zeros((len(E_g), len(E_i)))

    print("Starting optimization...")

    # Prepare argument list for parallel execution
    args_list = []
    for i in range(len(E_i)):
        for j in range(len(E_g)):
            args_list.append((i, j, E_g[j], E_i[i], int_max, T_cell, T_sun, fX))

    # Parallel execution
    with ThreadPoolExecutor() as executor:
        for i, j, eff in executor.map(optimize_point, args_list):
            efficiencies[i, j] = eff

    print(f"Max efficiency: {np.max(efficiencies):.2f}%")
    
    # Plotting with contour lines
    plt.figure(figsize=(10, 10))
    
    # Create the color mesh
    mesh = plt.pcolormesh(E_g, E_i, efficiencies, shading='auto', cmap='hot')
    
    plt.colorbar(mesh, label='Efficiency (%)')
    plt.xlabel('E_g (eV)')
    plt.ylabel('E_i (eV)')
    plt.title('Intermediate Band Solar Cell Efficiency Map with Contour Lines')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
