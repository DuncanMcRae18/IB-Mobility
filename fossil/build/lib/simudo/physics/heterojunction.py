import attr
import dolfin
import numpy as np

from .poisson_drift_diffusion import Band, NondegenerateBand
from ..mesh.topology import FacetRegion
from ..fem import expm1


@attr.s
class ThermionicHeterojunction:
    """
    Thermionic emission heterojunction BC valid with parabolic bands. 
    Treats both nondegenerate (Boltzmann) and degenerate bands

    Parameters
    ----------
    band: :py:class:`.Band`
        Semiconductor band on which the BC is applied. Only
        MixedQflBand is currently supported.
    boundary: :py:class:`.FacetRegion`
        Boundary on which to implement heterojunction boundary condition.

    Notes
    -----

    See V. Palankovski (2004), eq. 3.72 and
    K. Yang, J. R. East, G. I. Haddad, Solid State Electronics v.36 (3) p.321-330 (1993)
    K. Horio, H. Yanai, IEEE Trans. Elec. Devices v.37(4) p.1093-1098 (1990)

    For degenerate conditions, see Sentaurus sdevice manual

    For a conduction band BC, band.spatial must have attribute
    "CB/vth" in the barrier region. Similarly for other bands.

    Unlimited carrier flow from low to barrier region can be resolved, but due
    to precision issues, cannot resolve Delta_w producing carrier flows from
    barrier to low region with Delta_w larger than
    ``|ln(1e-16)| * kT`` in double precision
    """

    band: Band = attr.ib()
    boundary: FacetRegion = attr.ib()
    interface_quadrature_degree: int = attr.ib(default=20)
    HJBC_enhancement: dolfin.Constant = attr.ib(default=dolfin.Constant(1.0))
    use_alternative_BC: bool = attr.ib(default=False)
    use_nondegen: bool = attr.ib(default=False)
    add_additional_BC_term: bool = attr.ib(default=False)

    @property
    def vth(self):
        return self.band.spatial.get(self.band.spatial_prefix + "vth")

    @property
    def N_eff(self):
        return self.band.spatial.get(self.band.spatial_prefix + "effective_density_of_states")

    def register(self, debug_output=False):
        """
        Apply boundary condition onto the band.
        """

        band = self.band
        mu = band.pdd.mesh_util
        U = mu.unit_registry

        
        # delta: correction for tunneling through the barrier. not implemented.
        delta = 0.0
        sign = band.sign
        # Thermal velocities
        # These should be calculated from DOS effective mass (Palakovski 3.75)
        vth = self.vth
        # Eb is barrier height. should be positive. Use its sign to determine which
        # side is the low and which side is the barrier
        Eb = -sign * (mu.pside(band.energy_level) - mu.mside(band.energy_level))

        # 29 June 2022 -JK
        # Articles don't actually treat the case of holes explicitly. Previous version of HJBC
        # may have been using the wrong sense of the boundary when holes are considered.
        # Sentaurus' manual (I think) uses a convention where the band offset 
        # Delta_E = E_2-E_1 >0, so side "2" is the barrier for electrons but side "1" 
        # is the barrier for holes.
        # We can fix that by calling side the 2 the "high" side instead of the "bar" side 
        # (which was the old notation), and still keeping "low" for the "1" side.
        # Eb = mu.pside(band.energy_level) - mu.mside(band.energy_level)
        # If Eb>0, then pside=high. Else, pside=low

        # 18 Nov 2022 - JK & DX
        # We have determined that the proper sign convention is that Delta_E=E_b>0
        # for both electrons and holes. So we should use the "low" and "bar" 
        # labels for the code.

        def Eb_fconditional(fun1, fun2):
            """Return a function conditional on the sign of Eb """

            def func(expr):
                u = expr.units
                return u * dolfin.conditional(
                    dolfin.gt(Eb.m, 0), fun1(expr).m_as(u), fun2(expr).m_as(u)
                )

            return func

        lowside = Eb_fconditional(mu.mside, mu.pside)
        barside = Eb_fconditional(mu.pside, mu.mside)

        kT = lowside(band.kT)  # T is assumed continuous
        qe = U.elementary_charge

        u_bar = barside(band.u)
        vth_bar = barside(vth)
        N_eff_bar = barside(self.N_eff)
        # j_band is current from low to hi side.
        # mu.n points out from whichever surface. So to get current
        # from lowside to barside, need lowside(mu.n)
        j_band = lowside(mu.dot(band.j, mu.n))

        #w0 = band.mixedqfl_base_w
        w0 = band.qfl #Use correct qfl, so ufl derivative may work better with the BC
        phiqfl = band.phiqfl
        
        ##### nondegenerate expression
        # Use the ln function down to where we can't resolve it anyway,
        # then use a linear extrapolation of the log, which doesn't give
        # nan for negative argument
        # Apply an arcsinh on that linear extrapolation, to avoid its
        # growing too large
        shift = j_band / (sign * qe * vth_bar * (1 + delta) * u_bar)
        shift = shift.m_as(U.dimensionless)
        argument = 1 + shift
        # Use ln1p [= ln(1+x)] if shift is close to 0, for better precision
        large_target = dolfin.conditional(
            dolfin.gt(mu.dless(mu.abs(shift)), 1e-6),
            mu.ln(argument),
            mu.ln1p(shift),
        )
        eps = dolfin.DOLFIN_EPS
        small_target = mu.asinh((argument - eps) / eps) + np.log(eps)
        # Target for Delta_w
        Delta_w_BC_nondegen = (
            kT / sign
            * dolfin.conditional(
                dolfin.gt(argument, eps), large_target, small_target
            )
        )
        
        
        ##### degenerate
        # Notes 17 June 2022 & 28 June 2022
        # See code/doc/heterojunction_boundary_condition.lyx
        # Following Sentaurus sdevice manual, Eqs 879...
        # Results for degenerate HJ are not cited to any paper
        # a = 2 # Parameter from the Sentaurus manual
                # Don't see any reason to use the a=2 from the manual.
                # Its reference (Schroeder94 p 167) doesn't support the factor
                # Reduces to above Nondegenerate formulation when a=1
        a = 1
        B = j_band / (-sign * qe * a * vth_bar * N_eff_bar)
        B = B.m_as("dimensionless")
        E_bar = barside(band.energy_level)
        
        # 28 June version
        eta_2 = sign * ((E_bar - barside(phiqfl))/kT)
        # Delta_w_BC = kT/sign * mu.ln(
        #     mu.exp(-B) + mu.exp(-(B + eta_2)) - mu.exp(-eta_2)
        # )
        
        Delta_w_BC_degen = kT/sign * (-B + 
                            #  mu.ln(1 + mu.exp(-eta_2)*(1 - mu.exp(B))))
                            # mu.ln1p(mu.exp(-eta_2)*(1 - mu.exp(B))))
                            mu.ln1p(mu.exp(-eta_2)*
                                    (-expm1(B))
                                ))
     


        if isinstance(band, NondegenerateBand):           

            Delta_w_BC = Delta_w_BC_nondegen

        else: # Degenerate band
            
            # determine whether to use degenerate HJBC
            use_nondegen = getattr(self, 'use_nondegen', False)
            if use_nondegen:
                import logging
                logging.info("Use nondegenerate HJBC")
                Delta_w_BC = Delta_w_BC_nondegen
            else:
                Delta_w_BC = Delta_w_BC_degen
                 

        # Actual Delta_w
        Delta_w = barside(w0) - lowside(w0)
        boundary = self.boundary
        
        # probe degenerate and nondegenerate Delta_w_BC
        # Delta_w_BC_degen_probe = mu.get_debug_facet_probe(Delta_w_BC_degen, boundary)
        # Delta_w_BC_nondegen_probe = mu.get_debug_facet_probe(Delta_w_BC_nondegen, boundary)
        # print(f"band: {band.name}, Delta_w_BC_degen: {Delta_w_BC_degen_probe()}")
        # print(f"band: {band.name}, Delta_w_BC_nondegen: {Delta_w_BC_degen_probe()}")

        dS0 = mu.region_oriented_dS(boundary, orient=False)[1]
        dS = dS0(metadata = {"quadrature_degree": self.interface_quadrature_degree})
        xi = band.mixedqfl_xi

        # Remove the standard current due to w-changes from the interfaces
        band.mixedqfl_drift_diffusion_heterojunction_facet_region |= (
            boundary.both()
        )


        if not self.use_alternative_BC:
            BC_term = (Delta_w - Delta_w_BC).m_as("eV") # we shall put in the correct dimension later
        else:
            # from ..fem import ufl_extra as ufle
            # use alternative boundary condition for degenerate band

            eta = sign * Delta_w / kT
            eta = eta.m_as(U.dimensionless)
            # see Daisy's note 20231116
            # exponetiate both sides
            
            # lam1 and lam2 terms for nondegenerate
            lam1_BC_nondegen = ((j_band / (sign * qe * vth_bar * (1 + delta) * u_bar)).m_as("dimensionless") + 1) # "shift + 1" as above
            lam1_nondegen = mu.exp(eta)
            lam2_BC_nondegen = 1 / lam1_BC_nondegen
            lam2_nondegen = mu.exp(-eta)
            
            
            # lam1 and lam2 terms for degenerate
            x = (1 + mu.exp(-1 * eta_2.m_as("dimensionless")) * (-expm1(B)))

            lam1_degen = mu.exp(eta)
            lam1_BC_degen = (
                x * mu.exp(-B)
            )

            lam2_degen = mu.exp(-eta)
            lam2_BC_degen = (
                1/x * mu.exp(B)
            )
            
            # determine whether to use degenerate HJBC 
            if isinstance(band, NondegenerateBand) or self.use_nondegen:
                lam1_BC = lam1_BC_nondegen
                lam1 = lam1_nondegen
                lam2_BC = lam2_BC_nondegen
                lam2 = lam2_nondegen
            else:
                lam1_BC = lam1_BC_degen
                lam1 = lam1_degen
                lam2_BC = lam2_BC_degen
                lam2 = lam2_degen
            
            
            BC_term_Delta_w = (Delta_w - Delta_w_BC).m_as("eV")
            BC_term_lam1 = lam1 - lam1_BC
            BC_term_cosh = (lam1 + lam2) - (lam1_BC + lam2_BC)
            BC_term_sinh = (lam1 - lam2) - (lam1_BC - lam2_BC)
            
            ##################################################
            # standard alternative BC
            #-------------------------------------------------
            BC_term = BC_term_lam1
            ##################################################
            
            
            ##################################################   
            # give HJBC enhancement a boost when lam1 is small
            #-------------------------------------------------
            HJBC_enhancement = self.HJBC_enhancement * dolfin.conditional(
                dolfin.gt(lam1, 1e-6),
                1,
                1e-6 / lam1
            )
            ################################################## 
            
            
            ################################################## 
            # for lam1 sufficiently small, use Delta_w
            # doesn't work well, because when lam1 is small, lam_BC is prone to go negative
            #-------------------------------------------------
            # C = dolfin.Constant(1e-6)
            # BC_term = dolfin.conditional(
            #     dolfin.gt(lam1, C),
            #     BC_term_lam1,
            #     BC_term_Delta_w
            # )
            ##################################################
            
            
            ################################################## 
            # for lam1 sufficiently small, ramp up Delta_w, linear punishment of negative
            #-------------------------------------------------
            # C = dolfin.Constant(1e-6)
            
            # # linear punishment of negative lam1
            # linear_neg_lam1_punish = lam1_BC * dolfin.Constant(1e10)
            
            # # ramp up Delta_w
            # import math
            # log10e = dolfin.Constant(math.log10(math.e))
            # ramp_up = log10e * (mu.ln(lam1) - mu.ln(C)) / (10 - log10e * mu.ln(C))
            # Delta_w_ramp = Delta_w_BC * ramp_up
            
            # additional_BC_term = dolfin.conditional(
            #     dolfin.gt(lam1, C),
            #     0,
            #     dolfin.conditional(
            #         dolfin.lt(lam1_BC, 0),
            #         linear_neg_lam1_punish,
            #         Delta_w_ramp
            #     )
            # )
            ##################################################
            
            
            ################################################## 
            # use sinh, linear punishment of negative lam1
            # doesn't work well, because when lam1 is small, lam_BC is prone to go negative
            #-------------------------------------------------
            # C = dolfin.Constant(1e-6)
            
            # # linear punishment of negative lam1
            # linear_neg_lam1_punish = lam1_BC * dolfin.Constant(1e10)
            
            # BC_term = dolfin.conditional(
            #     dolfin.lt(lam1_BC, 0),
            #     linear_neg_lam1_punish,
            #     BC_term_sinh
            # )
            ##################################################
            
        # add boundary condition
        HJBC_enhancement = self.HJBC_enhancement  
        band.mixedqfl_drift_diffusion_heterojunction_bc_term += (
            -1 * dS * HJBC_enhancement * BC_term * U("eV") * lowside(mu.dot(xi, mu.n))
        )
        
        # add additional BC term if needed
        if self.add_additional_BC_term:
            band.mixedqfl_drift_diffusion_heterojunction_bc_term += (
                -1 * dS * HJBC_enhancement * additional_BC_term * U("eV") * lowside(mu.dot(xi, mu.n))
            )
        
        #debug code
        if hasattr(band.pdd, "debug_HJBC_reltol"):
            # # Delta_w_BC
            # debug_output_name = f"Delta_w_BC_{junction}"
            # Delta_w_BC_probe = mu.get_debug_facet_probe(Delta_w_BC, boundary)
            # setattr(band, debug_output_name, Delta_w_BC_probe)
            
            # # Delta_w
            # debug_output_name = f"Delta_w_{junction}"
            # Delta_w_probe = mu.get_debug_facet_probe(Delta_w, boundary)
            # setattr(band, debug_output_name, Delta_w_probe)
            
            # # j_band
            # debug_output_name = f"j_band_{junction}"
            # j_band_probe = mu.get_debug_facet_probe(j_band, boundary)
            # setattr(band, debug_output_name, j_band_probe)

            if not hasattr(band, 'Delta_w_BC'):
                band.Delta_w_BC = []
            if not hasattr(band, 'Delta_w_BC_degen'):
                band.Delta_w_BC_degen = []
            if not hasattr(band, 'Delta_w'):
                band.Delta_w = []
            if not hasattr(band, 'j_band'):
                band.j_band = []
            if not hasattr(band, 'phiqfl_facet'):
                band.phiqfl_facet = []
                
            if self.use_alternative_BC:
                if not hasattr(band, 'lam1'):
                    band.lam1 = []
                if not hasattr(band, 'lam1_BC'):
                    band.lam1_BC = []
                if not hasattr(band, 'lam2'):
                    band.lam2 = []
                if not hasattr(band, 'lam2_BC'):
                    band.lam2_BC = []

            Delta_w_BC_probe = mu.get_debug_facet_probe(Delta_w_BC, boundary)
            band.Delta_w_BC.append(Delta_w_BC_probe)
            
            Delta_w_BC_degen_probe = mu.get_debug_facet_probe(Delta_w_BC_degen, boundary)
            band.Delta_w_BC_degen.append(Delta_w_BC_degen_probe)

            Delta_w_probe = mu.get_debug_facet_probe(Delta_w, boundary)
            band.Delta_w.append(Delta_w_probe)

            j_band_probe = mu.get_debug_facet_probe(j_band, boundary)
            band.j_band.append(j_band_probe)

            phiqfl_facet_probe = mu.get_debug_facet_probe(barside(phiqfl), boundary)
            band.phiqfl_facet.append(phiqfl_facet_probe)
            
            if self.use_alternative_BC:
            
                lam1_probe = mu.get_debug_facet_probe(lam1, boundary)
                band.lam1.append(lam1_probe)

                lam1_BC_probe = mu.get_debug_facet_probe(lam1_BC, boundary)
                band.lam1_BC.append(lam1_BC_probe)

                lam2_probe = mu.get_debug_facet_probe(lam2, boundary)
                band.lam2.append(lam2_probe)

                lam2_BC_probe = mu.get_debug_facet_probe(lam2_BC, boundary)
                band.lam2_BC.append(lam2_BC_probe)