import pint
import os

"""this file is to demonstrate the problem that Fatemeh had with her versions. If run
with the /home/duncan/data/repo/envs/fatemeh_env_full.yml environment, with version 0.9
and 0.10.1 of pint it will print two different values for the permiability of free space."""

constants_file = "/home/duncan/Documents/School/Current Semester/Physics Project/Main/test/fossil/build/lib/simudo/util/pint/constants_en.txt"

print(f"Testing Pint {pint.__version__}")


u = pint.UnitRegistry()
u.load_definitions(constants_file)

mu_0 = u("vacuum_permeability")
epsilon_0 = u("vacuum_permittivity")

rho = 1.0 * u("C / cm**3")
epsilon_r = 13
permittivity = epsilon_r * epsilon_0

term = rho / permittivity

print("Raw Poisson Term (rho / permittivity):")
print(f"    {term.magnitude:.5e} {term.units}\n")

# 6. The FEniCS Handover (The SWIG float conversion)
converted = term.to("V / cm**2")

fenics_float = float(converted.magnitude) 
print(f"Output: {fenics_float:.5e}")

