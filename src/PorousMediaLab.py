import numpy as np
from scipy.sparse import spdiags
import Plotter
from Lab import Lab
import DESolver
import EquilibriumSolver
import pHcalc
from DotDict import DotDict


class PorousMediaLab(Lab):
    """PorousMediaLab module solves Advection-Diffusion-Reaction Equation in porous media"""

    def __init__(self, length, dx, tend, dt, phi, w=0):
        # ne.set_num_threads(ne.detect_number_of_cores())
        super().__init__(tend, dt)
        self.x = np.linspace(0, length, length / dx + 1)
        self.N = self.x.size
        self.length = length
        self.dx = dx
        self.w = w
        self.phi = np.ones((self.N)) * phi
        self.constants['phi'] = self.phi

    def add_temperature(self, init_temperature, D=281000):
        self.species['Temperature'] = DotDict({})
        self.species['Temperature']['is_solute'] = True
        self.species['Temperature']['bc_top_value'] = init_temperature
        self.species['Temperature']['bc_top_type'] = 'dirichlet'
        self.species['Temperature']['bc_bot_value'] = 0
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
        self.species[element]['bc_top_value'] = bc_top
        self.species[element]['bc_top_type'] = bc_top_type.lower()
        self.species[element]['bc_bot_value'] = bc_bot
        self.species[element]['bc_bot_type'] = bc_bot_type.lower()
        self.species[element]['theta'] = self.phi if is_solute else (
            1 - self.phi)
        self.species[element]['D'] = D
        self.species[element]['init_C'] = init_C
        self.species[element]['concentration'] = np.zeros(
            (self.N, self.time.size))
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
        self.species[element]['bc_top_value'] = bc

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

    def template_AL_AR(self, element):
        self.species[element]['AL'], self.species[element]['AR'] = DESolver.create_template_AL_AR(
            self.species[element]['theta'], self.species[element]['D'], self.species[element]['w'],
            self.species[element]['bc_top_type'], self.species[element]['bc_bot_type'], self.dt, self.dx, self.N)

    def update_matrices_due_to_bc(self, element, i):
        self.profiles[element], self.species[element]['B'] = DESolver.update_matrices_due_to_bc(self.species[
            element]['AR'], self.profiles[element], self.species[element]['theta'], self.species[element]['D'],
            self.species[element]['w'], self.species[element]['bc_top_type'], self.species[element]['bc_top_value'],
            self.species[element]['bc_bot_type'], self.species[element]['bc_bot_value'], self.dt, self.dx, self.N)

        self.species[element]['concentration'][:, i] = self.profiles[element]

    def create_acid_base_system(self):
        self.add_species(is_solute=True, element='pH', D=0, init_C=7, bc_top=7, bc_top_type='dirichlet',
                         bc_bot=7, bc_bot_type='dirichlet', rising_velocity=False, int_transport=False)
        self.acid_base_system = pHcalc.System(
            *[c['pH_object'] for c in self.acid_base_components])

    def acid_base_update_concentrations(self, i):
        for component in self.acid_base_components:
            init_conc = 0
            alphas = component['pH_object'].alpha(
                self.species['pH']['concentration'][:, i])
            for idx in range(len(component['species'])):
                init_conc += self.species[component['species']
                                          [idx]]['concentration'][:, i]
            for idx in range(len(component['species'])):
                self.species[component['species'][idx]
                             ]['concentration'][:, i] = init_conc * alphas[:, idx]
                self.profiles[component['species'][idx]] = self.species[component['species'][idx]]['concentration'][:, i]

    def integrate_one_timestep(self, i):
        if i == 1:
            self.pre_run_methods()
        self.transport_integrate(i)
        if self.rates:
            self.reactions_integrate(i)
        if self.henry_law_equations:
            self.henry_equilibrium_integrate(i)
        if self.acid_base_components:
            self.acid_base_equilibrium_solve(i)

    def transport_integrate(self, i):
        """ Integrates transport equations
        """
        for element in self.species:
            if self.species[element]['int_transport']:
                self.transport_integrate_one_element(element, i)

    def transport_integrate_one_element(self, element, i):
        self.profiles[element] = DESolver.linear_alg_solver(
            self.species[element]['AL'], self.species[element]['B'])
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
            flux = D * (-25 * self.phi[1] * C[1, idx] + 48 * self.phi[2] * C[2, idx] - 36 * self.phi[3] * C[
                3, idx] + 16 * self.phi[4] * C[4, idx] - 3 * self.phi[5] * C[5, idx]) / self.dx / 12 \
                - self.phi[0] * self.species[elem]['w'] * C[0]
        if order == 3:
            flux = D * (-11 * self.phi[1] * C[1, idx] + 18 * self.phi[2] * C[2, idx] -
                        9 * self.phi[3] * C[3, idx] + 2 * self.phi[4] * C[4, idx]) / self.dx / 6 \
                - self.phi[0] * self.species[elem]['w'] * C[0]
        if order == 2:
            flux = D * (-3 * self.phi[1] * C[1, idx] + 4 *
                        self.phi[2] * C[2, idx] * self.phi[3] * C[3, idx]) / self.dx / 2 \
                - self.phi[0] * self.species[elem]['w'] * C[0]
        if order == 1:
            flux = - D * (self.phi[0] * C[0, idx] * self.phi[2] * C[2, idx]) / 2 / self.dx \
                - self.phi[0] * self.species[elem]['w'] * C[0]

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
            flux = D * (-25 * self.phi[-2] * C[-2, idx] + 48 * self.phi[-3] * C[-3, idx] - 36 *
                        self.phi[-4] * C[-4, idx] + 16 * self.phi[-5] * C[-5, idx] - 3 * self.phi[-6] * C[-6, idx]) / self.dx / 12 \
                + self.phi[-1] * self.species[elem]['w'] * C[0]
        if order == 3:
            flux = D * (-11 * self.phi[-2] * C[-2, idx] + 18 * self.phi[-3] * C[-3, idx] -
                        9 * self.phi[-4] * C[-4, idx] + 2 * self.phi[-5] * C[-5, idx]) / self.dx / 6 \
                + self.phi[-1] * self.species[elem]['w'] * C[0]
        if order == 2:
            flux = D * (-3 * self.phi[-2] * C[-2, idx] + 4 *
                        self.phi[-3] * C[-3, idx] * self.phi[-4] * C[-4, idx]) / self.dx / 2 \
                + self.phi[-1] * self.species[elem]['w'] * C[0]
        if order == 1:
            flux = - D * (self.phi[-1] * C[-1, idx] * self.phi[-3] * C[-3, :]) / 2 / self.dx \
                + self.phi[-1] * self.species[elem]['w'] * C[0]

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
    plot_saturation_index = Plotter.saturation_index_countour
