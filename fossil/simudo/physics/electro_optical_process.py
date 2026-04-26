
from cached_property import cached_property

import dolfin
import ufl

import logging

from ..fem import expm1
from ..util import SetattrInitMixin

__all__ = [
    'ElectroOpticalProcess',

    'NonOverlappingTopHatBeerLambert',
    'NonOverlappingTopHatBeerLambertIB',

    'SRHRecombination',
    'NonRadiativeTrap',
    'ShockleyReadBand2BandTrap',

    'DarkEOPMixin',
    'TrapEOPMixin',
    'TwoBandEOPMixin',
    'RadiativeEOPMixin',
    'AbsorptionAndRadiativeRecombinationEOPMixin',
]

class ElectroOpticalProcess(SetattrInitMixin):
    '''This class exists to represent both electro-optical process and
purely electronic (dark) processes, such as nonradiative
recombination.

Attributes
----------
pdd: PoissonDriftDiffusion
    Instance of :class:`.PoissonDriftDiffusion`.
optical: Optical
    Instance of :class:`.Optical`.
pre_iteration_hook: callable
    Implement this method in order to run code before each PDD Newton
    iteration. Useful for processes that need procedural code (cannot
    be expressed as UFL form).
pre_first_iteration_hook: callable
    This method is called before the *first* PDD Newton iteration.
post_iteration_hook: callable
    This method is called after each PDD Newton iteration.
'''
    pre_iteration_hook = None
    pre_first_iteration_hook = None
    post_iteration_hook = None

    @property
    def unit_registry(self):
        return self.mesh_util.unit_registry

    @property
    def mesh_util(self):
        return self.pdd.mesh_util

    def get_optical_generation_by_optical_field(self, optical_field):
        '''Exists as a separate method to make anisotropy possible.

By default just calls :code:`self.get_optical_generation`, and
multiplies by :py:attr:`~.optical.OpticalField.solid_angle`.'''

        return (self.get_optical_generation(optical_field.photon_energy) *
                optical_field.solid_angle)

    def get_optical_generation(self, photon_energy):
        ''' Override for luminescent coupling. By default returns zero. '''
        return 0 * self.unit_registry("1/cm^3/s")

    def get_alpha_by_optical_field(self, optical_field):
        '''Exists as a separate method to make anisotropy possible.

        By default just calls ``self.get_alpha``.'''

        return self.get_alpha(optical_field.photon_energy)

    def get_alpha(self, photon_energy):
        '''This method is called to get Beer-Lambert absorption
coefficient alpha at a particular photon energy. Must return a UFL
quantity on the PDD mesh, or :code:`None`.

**User class must implement this.**
        '''
        raise NotImplementedError()

    def get_quantum_yield_by_optical_field(self, optical_field):
        '''Exists as a separate method to make anisotropy possible.

        By default just calls ``self.get_quantum_yield``.'''
        return self.get_quantum_yield(optical_field.photon_energy)

    def get_quantum_yield(self, photon_energy):
        ''' Default ``quantum_yield=1``. '''
        return 1.0

    def get_band_generation_sign(self, band):
        '''As the generation process intensifies (e.g., light
intensity increases), does :code:`band` lose or gain more carriers?

The default implementation always returns :code:`None`. User classes
are expected to override this method.

Note: If you are implementing a two-band process, you almost certainly
want to inherit from :py:class:`TwoBandEOPMixin`.

Returns
-------
sign: object
    If the band participates in the process, returns :code:`+1` or
    :code:`-1` if it gains or loses carriers,
    respectively. Otherwise, return :code:`None`.'''
        return None

    @cached_property
    def _zero_generation(self):
        '''Constant for zero generation, with the right units.'''
        return self.unit_registry("1/mesh_unit^3/s") * 0.0

    def get_generation(self, band):
        return self.get_generation_user(band)

    def get_generation_user(self, band):
        '''This method is called to get the generation contribution to
band :code:`band`. Must return a UFL quantity on the PDD mesh.

**User class may override this.** By default, this method just calls
:py:meth:`get_generation_optical`.

Sign convention: **The returned value is positive or negative if
the band gains or loses carriers respectively.**

This method will be called for *every* band in the system. You are strongly
encouraged to use :py:meth:`get_band_generation_sign` to determine
whether the process causes a gain or a loss of carriers in band
:code:`band`, and whether the band participates in the process at all.
'''

        return self.get_generation_optical(band)

    def get_generation_optical(self, band):
        '''Compute carrier generation due to optical absorption, using
absorption coefficient given by :py:meth:`get_alpha_by_optical_field`
and quantum yield given by :py:meth:`get_quantum_yield`.'''

        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation

        g = self._zero_generation
        for field in self.optical.fields:
            Phi = self.get_photon_flux_on_pdd_mesh(field)
            alpha = self.get_alpha_by_optical_field(field)
            
            # verbose = getattr(self, 'verbose', False)
            # if verbose:
            #     mu = self.mesh_util
            #     probe_y = 0.5
            #     probe_x = [0.01, 0.2, 0.5]
                
            #     # Phi
            #     # debug_probe_Phi = mu.get_debug_probe(Phi, "DG1")

            #     # alpha
            #     debug_probe_alpha = mu.get_debug_probe(alpha, "CG1")
                
            #     for xx in probe_x:
            #         # logging.info(f"{self.name} | Phi({xx})  | {debug_probe_Phi(xx, probe_y).to('1/cm^2/s')}")
            #         logging.info(f"{self.name} | abs alpha({xx})| {debug_probe_alpha(xx, probe_y).to('1/cm')} / cm")
                
                
            
            if alpha is not None:
                g = g + Phi*alpha*self.get_quantum_yield(field)*sign

        return g

    def get_photon_flux_on_pdd_mesh(self, optical_field):
        '''Returns optical photon flux, adapted onto PDD mesh. Use this
accessor instead of reaching inside :py:class:`.optical.OpticalField`
yourself.

Parameters
----------
optical_field: :py:class:`.optical.OpticalField`
    Optical field whose photon flux (clipped to be nonnegative) to get.
'''
        return optical_field.Phi_pddproj_clipped


class DarkEOPMixin():
    def get_alpha(self, photon_energy):
        return None


class TwoBandEOPMixin():
    '''Convenience mixin representing a generation process across
two bands.

Sign convention: **As the generation process intensifies, the
destination band gains more carriers.** The source band may gain or
lose carriers depending on its carrier type.

Attributes
----------
src_band: Band
    Source band (which may gain or lose a carrier depending on the sign).
dst_band: Band
    Destination band (which always gains a carrier).
'''

    def get_band_generation_sign(self, band):
        '''Provides a non-trivial implementation of
:py:meth:`ElectroOpticalProcess.get_band_generation_sign()` in the
case of two active bands.

See :py:class:`TwoBandEOPMixin` for the sign convention.'''
        if band == self.dst_band:
            return +1 # destination band gains carriers
        elif band == self.src_band:
            # what happens to source band depends on bands' carrier types
            return -(self.dst_band.sign * self.src_band.sign)
        else: # irrelevant band
            return None


class TrapEOPMixin():
    ''' Calculates properties involved in trapping, such as u1, tau, etc.

Attributes
----------
trap_band: Band
    Band doing the trapping.

reg_band: Band
    Non-trap band. e.g. CB or VB.'''

    @classmethod
    def easy_add_two_traps_to_pdd(
            cls, pdd, name_prefix, top_band, bottom_band, trap_band, **kwargs):

        CB = top_band
        VB = bottom_band
        IB = trap_band

        pdd.easy_add_electro_optical_process(
            cls, dst_band=CB, src_band=IB, trap_band=IB,
            name=name_prefix + '_top', **kwargs)
        pdd.easy_add_electro_optical_process(
            cls, dst_band=IB, src_band=VB, trap_band=IB,
            name=name_prefix + '_bottom', **kwargs)

    @cached_property
    def reg_band(self):
        if self.dst_band == self.trap_band:
            return self.src_band
        elif self.src_band == self.trap_band:
            return self.dst_band
        else:
            return None

    def get_trap_process_name(self, band):
        if band is None:
            name = self.name
        elif band == self.src_band:
            name = self.name + '_top'
        elif band == self.dst_band:
            name = self.name + '_bottom'
        else:
            raise ValueError()
        return name

    def trap_spatial_get(self, name, band=None):
        pname = self.get_trap_process_name(band)
        return self.pdd.spatial.get('/'.join((pname, name)))

    def get_trap_concentration(self, band=None):
        trap_band = self.trap_band
        if trap_band is None: # pull from spatial parameters
            return self.pdd.spatial.get('/'.join((self.name, 'N_t')))
        else: # just use trap band number of states
            return trap_band.number_of_states

    def get_tau(self, band=None):
        N_t = self.get_trap_concentration(band)
        c = self.get_capture_coefficient(band)
        return 1/(c * N_t)

    def get_sigma_th(self, band=None):
        return self.trap_spatial_get('sigma_th', band=band)

    def get_v_th(self, band=None):
        return self.trap_spatial_get('v_th', band=band)

    def get_capture_coefficient(self, band=None):
        '''See [Shockley1952a] (3.5).'''
        sigma_th = self.get_sigma_th(band)
        v_th = self.get_v_th(band)
        return sigma_th * v_th

    def get_trap_energy_level(self):
        trap_band = self.trap_band
        if trap_band is None:
            return self.pdd.spatial.get('/'.join((self.name, 'energy_level')))
        else:
            return trap_band.effective_energy_level # handle trap degeneracy

    def get_u1(self, band):
        E_I = self.get_trap_energy_level()        
        if band.is_degenerate_band:
            u1 = band.phiqfl_to_u_nondegenerate(E_I)
        else:
            u1 = band.phiqfl_to_u(E_I)
        return u1

    def get_shockley_read_trap_generation(self, capture_coeff):

        IB = self.trap_band
        RB = self.reg_band

        if IB.sign == RB.sign:
            # same signs for trap and regular band, need other type of
            # carrier in the trap because we're look for empty states
            # to get trapped in
            trap_filling_factor = IB.number_of_states - IB.u
        else:
            # opposite signs, just return trap carrier concentration
            trap_filling_factor = IB.u

        kT = self.pdd.kT
        x = (RB.sign * (RB.qfl - IB.qfl)/kT).m_as('dimensionless')
        r = (-expm1(x)) * RB.u * trap_filling_factor * capture_coeff

        # `-r` corresponds to the generation rate in the reg_band.
        # However, the sign convention for generation in
        # `TwoBandEOPMixin` requires the destination band to always gain
        # carriers as the process intensifies (for generation
        # processes).
        if self.dst_band != RB:
            r = r * -(self.dst_band.sign * RB.sign)

        return -r


class RadiativeEOPMixin():
    ''' Calculates properties involved in radiative processes, such as
trapping or recombination.

Only works in non-degenerate condition
:code:`exp((E_photon_min - mu_fi)/kT) >> 1`. See [Strandberg2011],
page 3, under Eq. 6.
'''

    def get_strandberg_I(self):
        u = self.unit_registry
        h = u('1 planck_constant')
        c = u('1 speed_of_light')
        kT = self.pdd.kT
        mu = self.pdd.mesh_util

        n_r = 1
        K = (8 * dolfin.pi * n_r **2) / (h**3 * c**2)

        E_L, E_U = self.get_absorption_bounds()
        
        # verbose = getattr(self, 'verbose', False)
        # if verbose:
        #     mu = self.mesh_util
        #     probe_y = 0.5
        #     probe_x = [0.01, 0.2, 0.5]
            
        #     # E_L
        #     debug_probe_E_L = mu.get_debug_probe(E_L, "DG1")
            
        #     # E_U
        #     debug_probe_E_U = mu.get_debug_probe(E_U, "DG1")
            
        #     for xx in probe_x:
        #         logging.info(f"{self.name} | E_L({xx}) | {debug_probe_E_L(xx, probe_y).to('eV')}")
        #         logging.info(f"{self.name} | E_U({xx}) | {debug_probe_E_U(xx, probe_y).to('eV')}")

        # equivalent to flipping integration bounds if E_U < E_L
        sign = dolfin.conditional(dolfin.gt(
            E_U.magnitude,
            E_L.m_as(E_U.units)), +1, -1)

        # antiderivative of `E^2 exp(E/kT)`
        def F(E):
            return -kT*(2*kT**2 + 2*E*kT + E**2) * mu.exp(-E/kT)

        I = F(E_U) - F(E_L)

        ans = K * I * sign

        # to evaluate integrals
        # ans = (ans.m((0.0, 0.0)) * ans.units).to("1 / centimeter ** 2 / second")
        # print(self.src_band.key, self.dst_band.key, ans)

        return ans

    def get_absorption_bounds(self):
        from .optical import AbsorptionRangesHelper
        ab = AbsorptionRangesHelper(problem_data=self.pdd.problem_data)

        keyname = frozenset((self.src_band.key, self.dst_band.key))

        return ab.get_transition_bounds().get(keyname)

    def inside_absorption_bounds_conditional(self, E, value):
        E_L, E_U = self.get_absorption_bounds()
        E_L = E_L.m_as('eV')
        E_U = E_U.m_as('eV')

        E = E.m_as('eV')
        return value.units * dolfin.conditional(
            ufl.And(E_L <= E, E_U > E), value.magnitude, 0)


class AbsorptionAndRadiativeRecombinationEOPMixin():
    enable_radiative_recombination = True

    def get_generation_user(self, band):
        '''Call :py:meth:`get_generation_optical` and
:py:meth:`get_radiative_recombination` and add together their results
accordingly.

Only include the radiative recombination process if
:py:attr:`enable_radiative_recombination` is true.
'''
        sign = self.get_band_generation_sign(band)
        
        if sign is None:
            return self._zero_generation

        g_abs = self.get_generation_optical(band) # absorption

        if self.enable_radiative_recombination:
            g_rad = self.get_radiative_recombination() * sign
        else:
            g_rad = self._zero_generation
            
        # probe the generations
        verbose = getattr(self, 'verbose', False)
        if verbose:
            mu = self.mesh_util
            probe_y = 0.5
            probe_x = [0.01, 0.2, 0.5]
            
            # g_rad
            debug_probe_rad = mu.get_debug_probe(g_rad, "DG1")
            
            # g_abs
            debug_probe_abs = mu.get_debug_probe(g_abs, "DG1")
            
            for xx in probe_x:
                logging.info(f"{self.name} | {band.name} | radiative ({xx}) | {debug_probe_rad(xx, probe_y).to('1/cm^3/s')}")
                logging.info(f"{self.name} | {band.name} | absorption({xx}) | {debug_probe_abs(xx, probe_y).to('1/cm^3/s')}")

        return g_abs - g_rad


class NonOverlappingTopHatBeerLambert(
        AbsorptionAndRadiativeRecombinationEOPMixin,
        RadiativeEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Absorption or radiative recombination between src_band and dst_band.
For absorption, src_band=VB and dst_band=CB, and vice versa for
recombination.

Includes radiative recombination by default.
'''

    name = 'beer_lambert'

    def get_constant_alpha(self):
        return self.pdd.spatial.get('/'.join((self.name, 'alpha')))

    def get_alpha(self, photon_energy):
        '''Return constant alpha, clipped by absorption bounds.'''
        return self.inside_absorption_bounds_conditional(
            photon_energy, self.get_constant_alpha())

    def get_radiative_recombination(self):
        alpha = self.get_constant_alpha()

        strb_I = self.get_strandberg_I()
        beta = 1/self.pdd.kT

        CB = self.dst_band
        VB = self.src_band
        
        x = (beta * (CB.qfl - VB.qfl)).m_as('dimensionless')

        # debug probes
        verbose = getattr(self, 'verbose', False)
        # if verbose:
        #     mu = self.mesh_util
            
        #     # alpha
        #     # debug_probe_alpha = mu.get_debug_probe(alpha, "CG1")
            
        #     # strb_I
        #     # debug_probe_strb_I = mu.get_debug_probe(strb_I, "DG1")
            
        #     # x
        #     # debug_probe_x = mu.get_debug_probe(x, "DG1")
            
        #     #expm1(x)
        #     # debug_probe_expm1x = mu.get_debug_probe(expm1(x), "DG1")
            
        #     probe_x = [0.01, 0.2, 0.5]
        #     probe_y = 0.5
        #     # for xx in probe_x:
        #         # logging.info(f"{self.name} | rad alpha({xx})      | {debug_probe_alpha(xx, probe_y).to('1/cm')}")
        #         # logging.info(f"{self.name} | strb_I({xx})    | {debug_probe_strb_I(xx, probe_y).to('1/cm^2/s')}")
        #         # logging.info(f"{self.name} | beta * delta_qfl ({xx})         | {debug_probe_x(xx, probe_y)} dimensionless")
        #         # logging.info(f"{self.name} | expm1(x)({xx})  | {debug_probe_expm1x(xx, probe_y)} dimensionless")

        return alpha * strb_I * expm1(x)
        # return alpha * strb_I * expm1x


class NonOverlappingTopHatBeerLambertIB(
        AbsorptionAndRadiativeRecombinationEOPMixin,
        RadiativeEOPMixin,
        TrapEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Absorption between intermediate band and regular band.

Includes radiative recombination by default.

Based on [Strandberg2011].'''

    name = 'beer_lambert_IB'

    def get_radiative_recombination(self):

        # pseudo capture coefficient
        capture_coeff = self.get_radiative_pseudo_capture_coefficient()
        g = self.get_shockley_read_trap_generation(capture_coeff)

        return -g

    def get_constant_sigma_opt(self):
        return self.pdd.spatial.get('/'.join((self.name, 'sigma_opt')))

    def get_sigma_opt(self, photon_energy):
        '''Return constant sigma_opt, clipped by absorption bounds.'''
        return self.inside_absorption_bounds_conditional(
            photon_energy, self.get_constant_sigma_opt())

    def get_alpha(self, photon_energy):

        IB = self.trap_band
        RB = self.reg_band

        sigma = self.get_sigma_opt(photon_energy)

        if IB.sign == RB.sign:
            # same signs for trap and regular band, the more carriers
            # in trap band the more this absorption process will
            # happen
            trap_filling_factor = IB.u
        else:
            # opposite signs, need complementary carrier type for trap
            trap_filling_factor = IB.number_of_states - IB.u

        return sigma*trap_filling_factor

    def get_radiative_pseudo_capture_coefficient(self):
        ''' See `eq:rad-u-form` in `doc/ib.lyx`. '''

        RB = self.reg_band

        sigma_opt = self.get_constant_sigma_opt()
        strandberg_I = self.get_strandberg_I()

        u1 = self.get_u1(RB)

        return sigma_opt * strandberg_I / u1


class SRHRecombination(
        DarkEOPMixin,
        TrapEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''SRH recombination. The destination band is the one *losing*
carriers through recombination.

Typically, dst_band=CB and src_band=VB.'''

    name = 'SRH'
    trap_band = None
    tau_from_spatial = True

    def get_tau(self, band):
        if self.tau_from_spatial:
            return self.pdd.spatial.get('/'.join((
                self.name, band.name, 'tau')))
        else:
            return super().get_tau(band)

    @cached_property
    def logger(self):
        return logging.getLogger("root")

    def get_generation_user(self, band):
        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation

        CB = self.dst_band
        VB = self.src_band

        if CB.sign == VB.sign:
            self.logger.warning('SRH recombination between two bands of the same sign is not physical')

        # When one of the bands doesn't exist in the full domain, these taus 
        # come back as 0 outside the proper domain. In those cases, we want 
        # r to be zero (as happens in the equivalent case for SR trapping), 
        # but the expression divides by zero. Put in a conditional to resolve.
        CB_tau = self.get_tau(CB)
        VB_tau = self.get_tau(VB)

        CB_u1 = self.get_u1(CB)
        VB_u1 = self.get_u1(VB)

        gamma_C, gamma_V = 1, 1
        if CB.is_degenerate_band:
            gamma_C = CB.u/CB.u_nondegenerate
        if VB.is_degenerate_band:
            gamma_V = VB.u/VB.u_nondegenerate

        # In the nondegenerate limit, we can use n_i^2 = CB.thermal_equilibrium_u * VB.thermal_equilibrium_u
        # But when at least one of the bands is degenerate, we need to use the Boltzmann definition of ni^2
        # Eg = CB.energy_level - VB.energy_level
        # kT = self.pdd.kT
        # mu = self.mesh_util
        # n_i_squared = CB.effective_density_of_states * VB.effective_density_of_states * mu.exp(-Eg/kT)
        n_i_squared = CB_u1 * VB_u1  #both calculated correctly with nondegenerate statistics

        rr = ((VB.u*CB.u - gamma_C*gamma_V*n_i_squared)/
             ((CB.u + gamma_C*CB_u1)*VB_tau +
              (VB.u + gamma_V*VB_u1)*CB_tau))

        # Return zero when tau's are all zero.
        mu = self.mesh_util
        r_units = rr.units
        r = dolfin.conditional( 
                dolfin.gt((mu.abs(CB_tau) + mu.abs(VB_tau)).m, 0),
                rr.m,
                0
            )*r_units

        g = -r

        return g * sign


class NonRadiativeTrap(
        DarkEOPMixin,
        TrapEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Shockley Read trapping process. Typically, trap_band = IB.
If you do not have an explicit :py:class:`Band` object for the trap
concentration, use :py:class:`SRHRecombination` instead.

See `doc/ib.lyx`.
'''

    name = 'nonradiative'

    def get_generation_user(self, band):
        g_sign = self.get_band_generation_sign(band)
        if g_sign is None:
            return self._zero_generation

        # capture coefficient
        capture_coeff = self.get_capture_coefficient()
        g = self.get_shockley_read_trap_generation(capture_coeff)

        return g * g_sign


class ShockleyReadBand2BandTrap(
        TwoBandEOPMixin,
        DarkEOPMixin,
        ElectroOpticalProcess):
    ''' Shockley-Read-like trapping process between the receiving (R) and sending (D) bands. 
    Similar to :py:class:`NonRadiativeTrap` (which is Shockley-Read trapping from a band to trap states with a single energy).

    If the signs of both bands are the same, s_D = s_R, then the rate is

        U = [1 - exp(s_D (w_D - w_R)/kT)] u_D/tau

    w_k is the qfl of band k
    u_D is the carrier concentration in band D
    where we approximate that the number of receiving states is independent of w_R
    Derivation assumes that the receiving band (R) is essentially entirely empty, 
    so only the population of the sending band (D) enters the rate. 

    The energy-averaged (usually phenomenological) capture time tau must be defined.
    Convention: only the value of tau in the **dst_band** will be used.
    If the instance of this EOP has :py:attr:`name` `SR_B2B` and the D band :py:attr:`name` is `DB`
    tau is found from `spatial["SR_B2B/DB/tau"]` 

     
    If s_D = -s_R, then the rate is

        U = [1 - exp(s_D (w_D - w_R)/kT)] u_D u_R <c>

    <c> is the average capture rate [volume^-1 time^-1] for the process.
    If the instance of this EOP has :py:attr:`name` `SR_B2B` and the D band :py:attr:`name` is `DB`
    <c> is found from `spatial["SR_B2B/DB/capture_rate"]`

    (In the previous case, 1/tau = <c> u_pR
    where u_pR is the concentration of carriers of opposite sign in band R)

    Following the standard sign convention, dst_band = D, src_band = R
    '''

    name = 'SR_B2B'

    def get_generation_user(self, band):
        g_sign = self.get_band_generation_sign(band)
        if g_sign is None:
            return self._zero_generation

        if self.dst_band.sign == self.src_band.sign:
            g = self.get_SR_generation_same_sign()
        else:
            g = self.get_SR_generation_diff_sign()

        return g * g_sign

    def get_SR_generation_same_sign(self):
        tau = self.pdd.spatial.get('/'.join((self.name, self.dst_band.name, "tau")))
        RB = self.src_band
        DB = self.dst_band

        kT = self.pdd.kT
        x = (DB.sign * (DB.qfl - RB.qfl)/kT).m_as("dimensionless")
        rr = -expm1(x) * DB.u / tau

        # Return zero when tau is zero.
        mu = self.mesh_util
        r_units = rr.units
        r = dolfin.conditional( 
                dolfin.gt(mu.abs(tau).m, 0),
                rr.m,
                0
            )*r_units

        return -r
        
    def get_SR_generation_diff_sign(self):
        c = self.pdd.spatial.get('/'.join((self.name, self.dst_band.name, "capture_rate")))
        RB = self.src_band
        DB = self.dst_band

        kT = self.pdd.kT
        x = (DB.sign * (DB.qfl - RB.qfl)/kT).m_as("dimensionless")
        r = -expm1(x) * DB.u * RB.u * c

        return -r