import numpy as np
from scipy.sparse import spdiags
import time
import sys
import traceback
import Plotter

import OdeSolver
import EquilibriumSolver
import pHcalc


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PorousMediaLab:
    """PorousMediaLab module solves Advection-Diffusion-Reaction Equation in porous media"""

    def __init__(self, length, dx, tend, dt, phi, w=0):
        # ne.set_num_threads(ne.detect_number_of_cores())
        self.x = np.linspace(0, length, length / dx + 1)
        self.time = np.linspace(0, tend, round(tend / dt) + 1)
        self.N = self.x.size
        self.length = length
        self.dx = dx
        self.tend = tend
        self.dt = dt
        self.phi = np.ones((self.N)) * phi
        self.w = w
        self.species = DotDict({})
        self.profiles = DotDict({})
        self.dcdt = DotDict({})
        self.rates = DotDict({})
        self.estimated_rates = DotDict({})
        self.constants = DotDict({})
        self.constants['phi'] = self.phi
        self.num_adjustments = 0
        self.henry_law_equations = []
        self.acid_base_components = []
        self.acid_base_system = pHcalc.System()

    def __getattr__(self, attr):
        return self.species[attr]

    def add_temperature(self, init_temperature, D=281000):
        self.species['Temperature'] = DotDict({})
        self.species['Temperature']['is_solute'] = True
        self.species['Temperature']['bc_top'] = init_temperature
        self.species['Temperature']['bc_top_type'] = 'dirichlet'
        self.species['Temperature']['bc_bot'] = 0
        self.species['Temperature']['bc_bot_type'] = 'neumann'
        # considering diffusion of temperature through pores (ie assumption is
        # that solid soil is not conducting heat)
        self.species['Temperature']['theta'] = self.phi
        self.species['Temperature']['D'] = D
        self.species['Temperature']['init_C'] = init_temperature
        self.species['Temperature']['concentration'] = np.zeros(
            (self.N, self.time.size))
        self.species['Temperature']['rates'] = np.zeros(
            (self.N, self.time.size))
        self.species['Temperature']['concentration'][
            :, 0] = (init_temperature * np.ones((self.N)))
        self.profiles['Temperature'] = self.species['Temperature']['concentration'][:, 0]
        self.species['Temperature']['w'] = 0
        self.species['Temperature']['int_transport'] = True
        self.template_AL_AR('Temperature')
        self.update_matrices_due_to_bc('Temperature', 0)
        self.dcdt['Temperature'] = '0'

    def add_species(self, is_solute, element, D, init_C, bc_top, bc_top_type, bc_bot, bc_bot_type, rising_velocity=False, int_transport=True):
        self.species[element] = DotDict({})
        self.species[element]['is_solute'] = is_solute
        self.species[element]['bc_top'] = bc_top
        self.species[element]['bc_top_type'] = bc_top_type.lower()
        self.species[element]['bc_bot'] = bc_bot
        self.species[element]['bc_bot_type'] = bc_bot_type.lower()
        self.species[element]['theta'] = self.phi if is_solute else (1 - self.phi)
        self.species[element]['D'] = D
        self.species[element]['init_C'] = init_C
        self.species[element]['concentration'] = np.zeros((self.N, self.time.size))
        self.species[element]['rates'] = np.zeros((self.N, self.time.size))
        self.species[element]['concentration'][:, 0] = self.species[element]['init_C']
        self.profiles[element] = self.species[element]['concentration'][:, 0]
        if rising_velocity:
            self.species[element]['w'] = self.w - rising_velocity
        else:
            self.species[element]['w'] = self.w
        self.species[element]['int_transport'] = int_transport
        if int_transport:
            self.template_AL_AR(element)
            self.update_matrices_due_to_bc(element, 0)
        self.dcdt[element] = '0'

    def add_solute_species(self, element, D, init_C):
        self.add_species(True, element, D, init_C, bc_top=0,
                         bc_top_type='neumann', bc_bot=0, bc_bot_type='neumann')

    def add_solid_species(self, element, init_C):
        self.add_species(False, element, 1e-18, init_C, bc_top=0,
                         bc_top_type='neumann', bc_bot=0, bc_bot_type='neumann')

    def new_top_boundary_condition(self, element, bc):
        self.species[element]['bc_top'] = bc

    def change_boundary_conditions(self, element, i, bc_top=False, bc_top_type=False, bc_bot=False, bc_bot_type=False):
        if bc_top_type is not False:
            self.species[element].bc_top_type = bc_top_type.lower()
        if bc_top is not False:
            self.species[element].bc_top = bc_top
        if bc_bot_type is not False:
            self.species[element].bc_bot_type = bc_bot_type.lower()
        if bc_bot is not False:
            self.species[element].bc_bot = bc_bot
        self.template_AL_AR(element)
        self.update_matrices_due_to_bc(element, i)

    def change_concentration_profile(self, element, i, new_profile):
        self.profiles[element] = new_profile
        self.update_matrices_due_to_bc(element, i)

    def template_AL_AR(self, element):
        # TODO: error source somewhere in non constant porosity profile. Maybe we also need d phi/dx
        s = self.species[element]['theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
        q = self.species[element]['theta'] * self.species[element]['w'] * self.dt / self.dx
        self.species[element]['AL'] = spdiags([-s / 2 + q / 4, self.species[element]['theta'] + s, -s / 2 - q / 4], [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[element]['AR'] = spdiags([s / 2 - q / 4, self.species[element]['theta'] - s, s / 2 + q / 4], [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()

        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant']:
            self.species[element]['AL'][0, 0] = self.species[element]['theta'][0]
            self.species[element]['AL'][0, 1] = 0
            self.species[element]['AR'][0, 0] = self.species[element]['theta'][0]
            self.species[element]['AR'][0, 1] = 0
        elif self.species[element]['bc_top_type'] in ['neumann', 'flux']:
            self.species[element]['AL'][0, 0] = self.species[element]['theta'][0] + s[0] + self.species[element]['w'] * s[0] * \
                self.dx / self.species[element]['D'] - q[0] * self.species[element]['w'] * self.dx / self.species[element]['D'] / 2
            self.species[element]['AL'][0, 1] = -s[1]  # / 2 - s[0] / 2
            self.species[element]['AR'][0, 0] = self.species[element]['theta'][0] - s[0] - self.species[element]['w'] * s[0] * \
                self.dx / self.species[element]['D'] + q[0] * self.species[element]['w'] * self.dx / self.species[element]['D'] / 2
            self.species[element]['AR'][0, 1] = s[1]  # / 2 + s[0] / 2
        else:
            print('\nABORT!!!: Not correct top boundary condition type...')
            sys.exit()

        if self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.species[element]['AL'][-1, -1] = self.species[element]['theta'][-1]
            self.species[element]['AL'][-1, -2] = 0
            self.species[element]['AR'][-1, -1] = self.species[element]['theta'][-1]
            self.species[element]['AR'][-1, -2] = 0
        elif self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['AL'][-1, -1] = self.species[element]['theta'][-1] + s[-1] + self.species[element]['w'] * s[-1] * \
                self.dx / self.species[element]['D'] - q[-1] * self.species[element]['w'] * self.dx / self.species[element]['D'] / 2
            self.species[element]['AL'][-1, -2] = -s[-2]  # / 2 - s[-1] / 2
            self.species[element]['AR'][-1, -1] = self.species[element]['theta'][-1] - s[-1] - self.species[element]['w'] * s[-1] * \
                self.dx / self.species[element]['D'] + q[-1] * self.species[element]['w'] * self.dx / self.species[element]['D'] / 2
            self.species[element]['AR'][-1, -2] = s[-2]  # / 2 + s[-1] / 2
        else:
            print('\nABORT!!!: Not correct bottom boundary condition type...')
            sys.exit()

    def update_matrices_due_to_bc(self, element, i):
        s = self.species[element]['theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
        q = self.species[element]['theta'] * self.species[element]['w'] * self.dt / self.dx

        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])

        elif self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * 2 * self.species[element]['bc_bot'] * \
                (s[-1] / 2 - q[-1] / 4) * self.dx / self.species[element]['theta'][-1] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * 2 * self.species[element]['bc_top'] * \
                (s[0] / 2 - q[0] / 4) * self.dx / self.species[element]['theta'][0] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * 2 * self.species[element]['bc_top'] * \
                (s[0] / 2 - q[0] / 4) * self.dx / self.species[element]['theta'][0] / self.species[element]['D']
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * 2 * self.species[element]['bc_bot'] * \
                (s[-1] / 2 - q[-1] / 4) * self.dx / self.species[element]['theta'][-1] / self.species[element]['D']
        else:
            print('\nABORT!!!: Not correct boundary condition in the species...')
            sys.exit()

        self.species[element]['concentration'][:, i] = self.profiles[element]

    def add_henry_law_equilibrium(self, aq, gas, Hcc):
        """Summary

        Args:
            aq (string): name of aquatic species
            gas (string): name of gaseous species
            Hcc (double): Henry Law Constant
        """
        self.henry_law_equations.append({'aq': aq, 'gas': gas, 'Hcc': Hcc})

    def add_ion(self, element, charge):
        ion = pHcalc.Neutral(charge=charge, conc=np.nan)
        self.acid_base_components.append({'species': [element], 'pH_object': ion})

    def add_acid(self, species, pKa, charge=0):
        acid = pHcalc.Acid(pKa=pKa, charge=charge, conc=np.nan)
        self.acid_base_components.append({'species': species, 'pH_object': acid})

    def acid_base_equilibrium_solve(self, i):
        self.acid_base_solve_ph(i)
        self.acid_base_update_concentrations(i)

    def acid_base_update_concentrations(self, i):
        for component in self.acid_base_components:
            init_conc = 0
            alphas = component['pH_object'].alpha(self.species['pH']['concentration'][:, i])
            for idx in range(len(component['species'])):
                init_conc += self.species[component['species'][idx]]['concentration'][:, i]
            for idx in range(len(component['species'])):
                self.species[component['species'][idx]]['concentration'][:, i] = init_conc * alphas[:, idx]

    def acid_base_solve_ph(self, i):
        res = self.species['pH']['concentration'][0, i - 1]  # initial guess from previous time-step
        for idx_j in range(self.N):
            for c in self.acid_base_components:
                init_conc = 0
                for element in c['species']:
                    init_conc += self.species[element]['concentration'][idx_j, i]
                c['pH_object'].conc = init_conc
            if idx_j == 0:
                self.acid_base_system.pHsolve(guess=7, tol=1e-4)
                res = self.acid_base_system.pH
            else:
                phs = np.linspace(res - 0.1, res + 0.1, 201)
                idx = self.acid_base_system._diff_pos_neg(phs).argmin()
                res = phs[idx]
            self.species['pH']['concentration'][idx_j, i] = res
            self.profiles['pH'][idx_j] = res

    def estimate_time_of_computation(self, i):
        if i == 1:
            self.tstart = time.time()
            print("Simulation started:\n\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if i == 100:
            total_t = len(self.time) * (time.time() - self.tstart) / 100 * self.dt / self.dt
            m, s = divmod(total_t, 60)
            h, m = divmod(m, 60)
            print("\n\nEstimated time of the code execution:\n\t %dh:%02dm:%02ds" % (h, m, s))
            print("Will finish approx.:\n\t", time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_t)))

    def create_acid_base_system(self):
        self.add_species(is_solute=True, element='pH', D=0, init_C=7, bc_top=7, bc_top_type='dirichlet',
                         bc_bot=7, bc_bot_type='dirichlet', rising_velocity=False, int_transport=False)
        self.acid_base_system = pHcalc.System(*[c['pH_object'] for c in self.acid_base_components])

    def pre_run_methods(self):
        if len(self.acid_base_components) > 0:
            self.create_acid_base_system()
            self.acid_base_equilibrium_solve(0)

    def solve(self, do_adjust=True):
        print('Simulation starts  with following params:\n\ttend = %.1f,\n\tdt = %.2e,\n\tL = %.1f,\n\tdx = %.2e,\n\tw = %.2f' %
              (self.time[-1], self.dt, self.length, self.dx, self.w))
        with np.errstate(invalid='raise'):
            for i in np.arange(1, len(np.linspace(0, self.tend, round(self.tend / self.dt) + 1))):
                try:
                    self.integrate_one_timestep(i)
                    self.estimate_time_of_computation(i)
                except FloatingPointError as inst:
                    print('\nABORT!!!: Numerical instability... Please, adjust dt and dx manually...')
                    traceback.print_exc()
                    sys.exit()

    def integrate_one_timestep(self, i):
        if i == 1:
            self.pre_run_methods()
        self.transport_integrate(i)
        self.reactions_integrate(i)
        if self.henry_law_equations:
            self.henry_equilibrium_integrate(i)
        if self.acid_base_components:
            self.acid_base_equilibrium_solve(i)

    def henry_equilibrium_integrate(self, i):
        for eq in self.henry_law_equations:
            self.species[eq['gas']]['concentration'][:, i], self.species[eq['aq']]['concentration'][:, i] = EquilibriumSolver.solve_henry_law(
                self.species[eq['aq']]['concentration'][:, i] + self.species[eq['gas']]['concentration'][:, i], eq['Hcc'])
            for elem in [eq['gas'], eq['aq']]:
                self.profiles[elem] = self.species[elem]['concentration'][:, i]
                if self.species[elem]['int_transport']:
                    self.update_matrices_due_to_bc(elem, i)

    def reactions_integrate(self, i):
        C_new, rates = OdeSolver.ode_integrate(
            self.profiles, self.dcdt, self.rates, self.constants, self.dt, solver='rk4')

        for element in C_new:
            if element is not 'Temperature':
                # the concentration should be positive
                C_new[element][C_new[element] < 0] = 0
            self.profiles[element] = C_new[element]
            self.species[element]['concentration'][:, i] = self.profiles[element]
            self.species[element]['rates'][:, i] = rates[element] / self.dt
            if self.species[element]['int_transport']:
                self.update_matrices_due_to_bc(element, i)

    def transport_integrate(self, i):
        """ Integrates transport equations
        """
        for element in self.species:
            if self.species[element]['int_transport']:
                self.transport_integrate_one_element(element, i)

    def transport_integrate_one_element(self, element, i):
        self.profiles[element] = OdeSolver.linear_alg_solver(self.species[element]['AL'], self.species[element]['B'])
        self.species[element]['concentration'][:, i] = self.profiles[element]
        self.update_matrices_due_to_bc(element, i)

    def estimate_flux_at_top(self, elem, idx=slice(None, None, None), order=4):
        """
        Function estimates flux at the top BC

        Args:
            elem (TYPE): name of the element
            order (int, optional): order of the Derivative

        Returns:
            TYPE: estimated flux in time

        """
        C = self.species[elem]['concentration']
        D = self.species[elem]['D']

        if order == 4:
            flux = D * (-25 * C[1, idx] + 48 * C[2, idx] - 36 * C[
                3, idx] + 16 * C[4, idx] - 3 * C[5, idx]) / self.dx / 12
        if order == 3:
            flux = D * (-11 * C[1, idx] + 18 * C[2, idx] -
                        9 * C[3, idx] + 2 * C[4, idx]) / self.dx / 6
        if order == 2:
            flux = D * (-3 * C[1, idx] + 4 *
                        C[2, idx] * C[3, idx]) / self.dx / 2
        if order == 1:
            flux = - D * (C[0, idx] * C[2, idx]) / 2 / self.dx

        return flux

    def estimate_flux_at_bottom(self, elem, idx=slice(None, None, None), order=4):
        """
        Function estimates flux at the bottom BC

        Args:
            elem (TYPE): name of the element
            order (int, optional): order of the Derivative

        Returns:
            TYPE: estimated flux in time

        """

        C = self.species[elem]['concentration']
        D = self.species[elem]['D']

        if order == 4:
            flux = D * (-25 * C[-2, idx] + 48 * C[-3, idx] - 36 *
                        C[-4, idx] + 16 * C[-5, idx] - 3 * C[-6, idx]) / self.dx / 12
        if order == 3:
            flux = D * (-11 * C[-2, idx] + 18 * C[-3, idx] -
                        9 * C[-4, idx] + 2 * C[-5, idx]) / self.dx / 6
        if order == 2:
            flux = D * (-3 * C[-2, idx] + 4 *
                        C[-3, idx] * C[-4, idx]) / self.dx / 2
        if order == 1:
            flux = - D * (C[-1, idx] * C[-3, :]) / 2 / self.dx

        return flux

    def is_solute(self, element):
        return self.species[element]['is_solute']

    """Mapping of plotting methods from Plotter module"""

    custom_plot = Plotter.custom_plot
    plot_depths = Plotter.plot_depths
    plot_times = Plotter.plot_times
    plot_profiles = Plotter.plot_profiles
    plot_profile = Plotter.plot_profile
    plot_contourplots = Plotter.plot_contourplots
    contour_plot = Plotter.contour_plot
    plot_contourplots_of_rates = Plotter.plot_contourplots_of_rates
    contour_plot_of_rates = Plotter.contour_plot_of_rates
