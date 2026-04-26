# import necessary libraries
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, minimize, root, root_scalar
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np
import warnings

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
    P_in = fX * (pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4')

    # return power
    return P_in  # Flux in photons/(m^2·s·m)

def photon_flux(min, max, T, mu, fX):
    # remove units from constants
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    # Add bounds checking to avoid integration issues
    if min >= max or min < 0:
        return u.Quantity(0, '1/(m^2*s)')
    
    # Suppress warnings and use more robust integration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            # Use more robust integration parameters
            flux, error = quad(
                lambda x: (x**2 / (np.exp((x - mu) / (k * T)) - 1)), 
                min, max,
                limit=200,      # More subdivisions
                epsabs=1e-8,    # Looser absolute tolerance
                epsrel=1e-6     # Looser relative tolerance
            )
        except:
            # If integration fails, return zero flux
            flux = 0

    # determine flux
    N = (fX * pi / ((h**3) * (c**2))) * flux * eV**3

    # return flux
    return N

def IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci):

    u_iv = u_cv - u_ci

    # if statememnt to choose limits of integration
    if E_i < E_c:
        flux_val = photon_flux(E_i, E_c, T_sun, 0 * eV, fX) - photon_flux(E_i, E_c, T_cell, u_iv, 1)
        flux_conduct = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci, 1)
    else:
        flux_val = photon_flux(E_i, E_g, T_sun, 0 * eV, fX) - photon_flux(E_i, E_g, T_cell, u_iv, 1)
        flux_conduct = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci, 1)

    # determine total flux
    flux = (flux_val - flux_conduct)

    # Convert to current density (same units as CB_current_density)
    J = q * flux
    
    # return current density (not just flux)
    return J

def CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv):

    # conduction flux is simple to calculate
    conduct_flux = photon_flux(E_g, int_max * eV, T_sun, 0 * eV, fX) - photon_flux(E_g, int_max * eV, T_cell, u_cv, 1)

    # if statement to choose limits of integration, then choose minimum flux to ensure current balance
    if E_i  <= E_c:
        inter_flux = photon_flux(E_c, E_g, T_sun, 0 * eV, fX) - photon_flux(E_c, E_g, T_cell, u_ci, 1)
    else:
        inter_flux = photon_flux(E_c, E_i, T_sun, 0 * eV, fX) - photon_flux(E_c, E_i, T_cell, u_ci, 1)

    # calculate total flux
    flux = conduct_flux + inter_flux

    # calculate current density
    J = q * flux

    # return current density
    return J

def find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv):
    def equation(u_ci_val):
        return IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci_val * eV)
    
    maximum = min(E_g.m_as('eV'), u_cv.m_as('eV'))
    
    # Generate points from two Gaussian distributions
    mean1 = 0.3 * maximum
    mean2 = 0.7 * maximum
    var = 0.08 * maximum
    
    # Sample points from each Gaussian
    n_samples = 30
    gauss1 = np.random.normal(mean1, var, n_samples)
    gauss2 = np.random.normal(mean2, var, n_samples)
    guess = np.concatenate([gauss1, gauss2])
    guess = np.clip(guess, 0.15 * maximum, 0.85 * maximum)

    current = None

    for i in guess:
        current = abs(equation(i).m_as('A/m^2'))
        if current < 5:
            break
    
    # Run root finder with best point as initial guess
    if current is not None:
        zero = root(equation, i, method='lm', tol=1e-3)
        if zero.success:
            return zero.x[0], True
    return -1, False

def efficiency(u_cv, E_g, E_i, T_cell, T_sun, int_max, fX, P_in):
    # simplify constants and variables
    E_c = E_g - E_i
    u_ci_val, valid = find_u_ci(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv)
    
    u_ci = u_ci_val * eV
    u_iv = u_cv - u_ci

    # check if u_ci was found
    if not valid or u_ci <= 0 or u_iv <= 0:
        return 0

    # check to ensure band gaps are valid
    if E_i <= 0 or E_c <= 0:  
        return 0

    if not (0.01 < u_cv.m_as('eV') < 0.99 * E_g.m_as('eV')):
        return 0
    if not (0.001 < u_ci.m_as('eV') < min(0.9 * E_c.m_as('eV'), 0.9 * u_cv.m_as('eV'))):
        return 0
    if not (0.01 < u_iv.m_as('eV') < min(0.99 * E_i.m_as('eV'), 0.99 * u_cv.m_as('eV'))):
        return 0

    try:
        # calculate output power and efficiency
        J = CB_current_density(E_g, E_c, E_i, int_max, T_cell, T_sun, fX, u_cv, u_ci, u_iv)
        P_out = J * u_cv / q
        efficiency = P_out / P_in

        # ensure that efficiency is within physical bounds
        if not 0 < efficiency < 1:
            return 0

        return -float(efficiency)
    except:
        return 0  # Large penalty for calculation errors

def Optimize_u_cv(E_i, E_g, int_max, T_cell, T_sun, fX):
    # Calculate power input once
    P_in = power_input(T_sun, int_max, fX)

    
    # set bounds for u_cv
    bounds = [
        (0.01 * E_g.m_as('eV'), 0.99 * E_g.m_as('eV')),   # u_cv bounds
    ]
    
    # FIXED: Properly formatted objective function
    def function(u_cv):
        return efficiency(u_cv * eV, E_g, E_i, T_cell, T_sun, int_max, fX, P_in)

    #minimization using differential evolution
    guess = differential_evolution(
        function,
        bounds=bounds,
        strategy='best1bin',   # More robust for global search
        maxiter=100,           # Increase iterations
        popsize=10,            # Increase population size
        tol=1e-1,              # Reduce tolerance for finer convergence
        polish=True,           # Enable local search after global
    )

    result = minimize(
        function,
        [0.5 * E_g.m_as('eV')],
        bounds=bounds, 
        tol=1e-3, 
        method='SLSQP',
        options={'ftol': 1e-3}
    )

    if result.success:
        return result.fun  # Return float not tuple
    else:
        return 0

def Optimize_E_i(E_g, int_max, T_cell, T_sun, fX):
    
    # simplify E_g without units 
    E_g_val = E_g.m_as('eV')

    # set bounds for E_i
    bounds = [
        (0.01 * E_g_val, 0.99 * E_g_val),   # E_i bounds
    ]
    
    # FIXED: Properly formatted objective function
    def function(E_i):
        return Optimize_u_cv(E_i * eV, E_g, int_max, T_cell, T_sun, fX)
    
    print(f"  Starting E_i optimization for E_g = {E_g_val:.2f} eV")
    
    #minimization using differential evolution
    guess = differential_evolution(
        function,
        bounds=bounds,
        strategy='best1bin',   # More robust for global search
        maxiter=100,           # Increase iterations
        popsize=20,            # Increase population size
        tol=1e-3,              # Reduce tolerance for finer convergence
        polish=True,           # Enable local search after global
    )

    result = minimize(
        function,  # Use the same properly formatted function
        guess.x,
        bounds=bounds, 
        tol=1e-2, 
        method='L-BFGS-B'
    )

    if result.success and result.fun < -1e-6:  # Must be negative efficiency
        print(f"SUCCESS: E_g={E_g.m_as('eV'):.2f}, efficiency={-result.fun*100:.2f}%, E_i={result.x[0]:.2f}")
        return result.fun, result.x[0]
    else:
        print(f"FAILED: E_g={E_g.m_as('eV'):.2f}, returning zeros")
        return 0, 0

"""This portion of this code defines the parameters you wish to use"""

cplx = 20 # complexity of the calculations, higher values increase precision but also computation time

int_max = 10 # designate maximum energy to integrate to

E_g = np.linspace(0.7, 4.5, cplx) * eV  # Focused on high-efficiency range

if __name__ == "__main__":
    
    # Comment out the main section
    
    E_i_1 = np.zeros(cplx)
    E_i_2 = np.zeros(cplx)

    # two different X-factors for two different plots
    X1 = 1
    X2 = 46200
    f = 1/46200
    
    # Run optimization for X=1
    fX1 = f * X1
    print("Running optimization for one sun")
    efficiencies_1 = np.zeros(cplx)
    
    for i in range(cplx):
        eff, temp = Optimize_E_i(E_g[i], int_max, T_cell, T_sun, fX1)
        efficiencies_1[i] = -eff  # Convert back to positive
        E_i_1[i] = temp
        print(f"Added to plot: E_g={E_g[i].m_as('eV'):.2f}, efficiency={efficiencies_1[i]*100:.2f}%, E_i={E_i_1[i]:.2f}")

    # Run optimization for X=46200
    fX2 = f * X2
    print("Running optimization for full concentration")
    efficiencies_2 = np.zeros(cplx)
    
    for i in range(cplx):
        eff, temp = Optimize_E_i(E_g[i], int_max, T_cell, T_sun, fX2)
        efficiencies_2[i] = -eff  # Convert back to positive
        E_i_2[i] = temp

    # Create a single figure with both results
    plt.figure(figsize=(12, 8))
    
    # Plot curves
    plt.plot(E_g.m_as('eV'), efficiencies_1 * 100, label='X=1', color='blue')
    plt.plot(E_g.m_as('eV'), efficiencies_2 * 100, label='X=46200', color='red')

    # Get E_g values for peak marking
    E_g_values = E_g.m_as('eV')
    
    # Find and mark peak efficiency for X=1
    max_index1 = np.argmax(efficiencies_1)
    max_eff1 = efficiencies_1[max_index1] * 100
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
    max_index2 = np.argmax(efficiencies_2)
    max_eff2 = efficiencies_2[max_index2] * 100
    max_eg2 = E_g_values[max_index2]
    plt.annotate(f'Peak: {max_eff2:.2f}%',
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
    
    # Add text annotation explaining the concentration factors
    plt.figtext(0.15, 0.02, 'X=1: One sun concentration\nX=46200: Maximum solar concentration', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.show()
    
"""    
    # Test values for E_g = 1, 2, 3 eV with fX = 1/46200 (one sun)
    print("Testing specific E_g values with fX = 1/46200 (one sun)")
    print("=" * 60)
    
    test_E_g_values = [1.0, 2.0, 3.0]  # eV
    fX_test = 1/46200  # One sun concentration factor
    
    for E_g_val in test_E_g_values:
        print(f"\nTesting E_g = {E_g_val} eV:")
        print("-" * 30)
        
        E_g_test = E_g_val * eV
        
        try:
            eff, E_i_opt = Optimize_E_i(E_g_test, int_max, T_cell, T_sun, fX_test)
            efficiency_percent = -eff * 100  # Convert back to positive percentage
            
            print(f"Final Result: Efficiency = {efficiency_percent:.2f}%, Optimal E_i = {E_i_opt:.2f} eV")
            
        except Exception as e:
            print(f"Error during optimization: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")"""
