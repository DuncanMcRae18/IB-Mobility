# import necessary libraries
from matplotlib import pyplot as plt
from scipy.constants import pi
from scipy.integrate import quad
from pint import UnitRegistry
import numpy as np
import pandas as pd

# Simplify unit registry name
u = UnitRegistry()

# Define constants as simple variables
h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt

"""This python code is intended to model the detailed balance limit of efficiency for a single junction solar cell
   The functions are as follows:
        photon_flux - calculates the blackbody photon flux given some bandgap E_g, temperature of the material, and the chemical potential of that material
        current_density - calculates the current density using the photon flux
        power_in - calculates the total power input from the sun
        power_output - calculates the maximum power output for a given bandgap energy
        export - exports the results to a CSV file
        """

def photon_flux(E_g, T, mu_cv,fX):
    """Calculate the photon flux for a given bandgap energy, temperature, chemical potential, and solid angle"""

    N = u.Quantity('A/(m^2)') # Create an empty variable to store flux values

    "prior to calculations using numpy, all values must have their units removed to avoid errors, then the units are reapplied afterwards"
    T = T.to('K').magnitude # Ensure temperature is in Kelvin
    E_g = E_g.to('eV').magnitude # Ensure bandgap energy is in eV
    mu_cv = mu_cv.to('eV').magnitude # remove units for calculation, ensuring eV are used
    k = u.Quantity('boltzmann_constant').to('eV/K').magnitude # calculate the denominator for the integrand and remove units for calculation, ensuring eV are used

    integrand, error = quad(lambda x:(x**2/(np.exp(((x - mu_cv) / (k * T)))-1)), E_g, 20) # Integrate the photon flux from E_g to a high value

    N = (fX *pi / ((h**3) * (c**2))) * integrand * eV**3 # Calculate the photon flux and reapply units

    return N  # Flux in photons/(m^2·s)

def current_density(E_g, incoming_photons, T_cell, mu_cv):
    """calculate the current density using the photon flux in and out of the cell"""

    emitted_photons = photon_flux(E_g, T_cell, mu_cv,1) # Photon flux from the cell, chemical potential is mu_cv,fX is 1 for solar cell

    J = q * (incoming_photons - emitted_photons) # calculate current density

    return J  # Current density in A/m^2

def power_input(T_sun,fX): # Calculate the total power input from the sun
    P_in = u.Quantity('W/(m^2)') # Create an empty variable to store power input values

    T_sun = T_sun.to('K').magnitude # Ensure temperature is in Kelvin
    k = u.Quantity('boltzmann_constant').to('eV/K').magnitude # set the boltzmann constant in eV/K and remove units for calculation

    integrand, error = quad(lambda x:(x**3/(np.exp((x / (k * T_sun)))-1)) # Integrate the power input from 0 to a high value
                            ,0 , Comp / 10)

    P_in = fX * (pi/((h**3)*(c**2))) * integrand * u.Quantity('eV^4') # Calculate the total power input from the sun

    return P_in  # Flux in photons/(m^2·s·m)

def power_output(E_g, T_sun, T_cell,fX):
    """Find the maximum power output for each bandgap energy E_g"""

    max_P = u.Quantity(np.zeros(len(E_g)), 'W/m^2').to_base_units() # create an empty array to store max power values

    for i in range(len(E_g)): # loop over each bandgap energy

        print("Bandgap: ", i, " of ", len(E_g)) #print progress to console

        mu_cv = np.linspace(0, E_g[i], Comp * 5) # create an array of chemical potentials from 0 to E_g

        P_out = u.Quantity(np.zeros(len(mu_cv)), 'W/m^2') # create an empty array to store power output values
        
        incoming_photons = photon_flux(E_g[i], T_sun, 0 * eV,fX) # Photon flux from the sun, chemical potential is 0,fX is the solid angle factor

        j = 1 # initialize index for while loop
        while P_out[j-1] >= P_out[j-2] and j < len(mu_cv): # loop until change in power output is 0

            P_out[j] = (mu_cv[j] / q) * current_density(E_g[i], incoming_photons, T_cell, mu_cv[j]) # calculate power output for each chemical potential
            j += 1 # increment index

        max_P[i] = P_out.max() # store the maximum power output for this bandgap energy

    return max_P # Maximum power output for each E_g

def export(E_g, efficiency, Export): # Export results to a CSV file
    df = pd.DataFrame({'Bandgap Energy (eV)': E_g.magnitude, 'Max Efficiency': efficiency.magnitude})
    df.to_csv('detailed_balance_limit_data.csv')

"""The last portion of this code defines the parameters you wish to use"""

Comp = 100 # complexity of the calculations, higher values increase precision but also computation time
E_g = u.Quantity(np.linspace(0.1, 6, Comp), 'eV') # array of bandgap energies to calculate over
T_sun = 6000 * u.kelvin # temperature of the sun
T_cell = 300 * u.kelvin # temperature of the solar cell
f = 1/(14000*np.pi) # solid angle factor
X = 1 # concentration factor
fX = f * X # combined solid angle and concentration factor
Export = False # change to true to export data to an excel file

"""Main problem solving"""
efficiencies = power_output(E_g, T_sun, T_cell,fX).to('W/m^2')/ power_input(T_sun,fX).to('W/m^2')
if (Export == True):
    export(E_g, efficiencies)

"""Plotting results"""
plt.figure(figsize=(12, 8))
plt.plot(E_g.magnitude,efficiencies)
plt.xlabel('Bandgap Energy (eV)')
plt.ylabel('Max Efficiency')
plt.title('Detailed Balance Limit: Maximum Efficiency vs Bandgap Energy')
plt.hlines(efficiencies.max().magnitude, xmin=0, xmax=5, colors='g', linestyles='dashed', label=f'Max Efficiency: {efficiencies.max().magnitude:.2%} at {E_g[efficiencies.argmax()].magnitude:.2f} eV')
plt.legend()
plt.grid()
plt.show()
