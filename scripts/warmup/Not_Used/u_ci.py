import numpy as np
import matplotlib.pyplot as plt
from pint import UnitRegistry
from scipy.integrate import quad
from scipy.optimize import curve_fit
import warnings

u = UnitRegistry()
h = u.planck_constant
c = u.speed_of_light
k = u.boltzmann_constant
q = u.elementary_charge
eV = u.electron_volt
pi = np.pi

def photon_flux(min, max, T, mu, fX):
    T = T.m_as('K')
    min = min.m_as('eV')
    max = max.m_as('eV')
    k = u.Quantity(1, 'boltzmann_constant').m_as('eV/K')
    mu = mu.m_as('eV')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux, error = quad(lambda x: (x**2 / (np.expm1((x - mu) / (k * T)))), min, max)
    N = (fX * 2 * pi / ((h**3) * (c**2))) * flux * eV**3
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
    return q * flux_conduct_abs - q * flux_val_abs

if __name__ == "__main__":
    T_cell = 300 * u.kelvin
    T_sun = 6000 * u.kelvin
    fX = 1

    E_g_list = [1, 2, 3]  # eV

    E_i_list = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, E_g_val in enumerate(E_g_list):
        ax = axes[idx]
        E_i_list = [0.2, 0.4, 0.6, 0.8] * E_g_val
        E_g = E_g_val * eV
        u_cv = 0.8 * E_g

        for E_i_val in E_i_list:
            E_i = E_i_val * eV
            E_c = E_g - E_i

            # Calculate u_ci range for this E_g and E_i
            E_c_val = E_c.m_as('eV')
            u_cv_val = u_cv.m_as('eV')
            E_i_val_ev = E_i.m_as('eV')
            maximum = min(E_c_val, u_cv_val)
            minimum = u_cv_val - min(u_cv_val, E_i_val_ev)
            u_ci_vals = np.linspace(minimum + 1e-5, maximum - 1e-5, 200)

            currents = []
            for u_ci_val in u_ci_vals:
                try:
                    J = IB_current_density(E_g, E_c, E_i, T_cell, T_sun, fX, u_cv, u_ci_val * eV)
                    currents.append(J.m_as('A/m^2'))
                except Exception:
                    currents.append(np.nan)

            ax.plot(u_ci_vals, currents, label=f"E_i={E_i_val} eV")

            print(f"E_g={E_g_val}, E_i={E_i_val}, E_c={E_c_val}, u_cv={u_cv_val}, u_ci range=({minimum}, {maximum})")

        ax.set_title(f"E_g = {E_g_val} eV")
        ax.set_xlabel("u_ci (eV)")
        if idx == 0:
            ax.set_ylabel("IB Current Density (A/m²)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()