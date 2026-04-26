from pint import UnitRegistry
import dolfin

u = UnitRegistry()

print(1 * u.pi*dolfin.pi)