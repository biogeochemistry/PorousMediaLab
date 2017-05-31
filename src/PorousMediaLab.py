import numpy as np
from scipy.sparse import spdiags
import time
import sys
from joblib import Parallel, delayed
import Ploter

import OdeSolver


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
        self.parallel = Parallel(n_jobs=4)

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
        self.profiles['Temperature'] = self.species[
            'Temperature']['concentration'][:, 0]
        self.template_AL_AR('Temperature')
        self.update_matrices_due_to_bc('Temperature', 0)
        self.dcdt['Temperature'] = '0'

    def add_species(self, is_solute, element, D, init_C, bc_top, bc_top_type, bc_bot, bc_bot_type):
        self.species[element] = DotDict({})
        self.species[element]['is_solute'] = is_solute
        self.species[element]['bc_top'] = bc_top
        self.species[element]['bc_top_type'] = bc_top_type.lower()
        self.species[element]['bc_bot'] = bc_bot
        self.species[element]['bc_bot_type'] = bc_bot_type.lower()
        self.species[element][
            'theta'] = self.phi if is_solute else (1 - self.phi)
        self.species[element]['D'] = D
        self.species[element]['init_C'] = init_C
        self.species[element]['concentration'] = np.zeros(
            (self.N, self.time.size))
        self.species[element]['rates'] = np.zeros((self.N, self.time.size))
        self.species[element]['concentration'][
            :, 0] = self.species[element]['init_C']
        self.profiles[element] = self.species[element]['concentration'][:, 0]
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

    def template_AL_AR(self, element):
        s = self.species[element][
            'theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
        q = self.species[element]['theta'] * \
            self.w * self.dt / self.dx
        self.species[element]['AL'] = spdiags(((-s / 2 - q / 4), (self.species[element][
                                              'theta'] + s), (-s / 2 + q / 4)), [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[element]['AR'] = spdiags(((s / 2 + q / 4), (self.species[element][
                                              'theta'] - s), (s / 2 - q / 4)), [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()

        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant']:
            self.species[element]['AL'][
                0, 0] = self.species[element]['theta'][0]
            self.species[element]['AL'][0, 1] = 0
            self.species[element]['AR'][
                0, 0] = self.species[element]['theta'][0]
            self.species[element]['AR'][0, 1] = 0
        elif self.species[element]['bc_top_type'] in ['neumann', 'flux']:
            self.species[element]['AL'][0, 0] = self.species[
                element]['theta'][0] + s[0]
            self.species[element]['AL'][0, 1] = -s[0]
            self.species[element]['AR'][0, 0] = self.species[
                element]['theta'][0] - s[0]
            self.species[element]['AR'][0, 1] = s[0]
        else:
            print('\nABORT!!!: Not correct top boundary condition type...')
            sys.exit()

        if self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.species[element]['AL'][-1, -
                                        1] = self.species[element]['theta'][-1]
            self.species[element]['AL'][-1, -2] = 0
            self.species[element]['AR'][-1, -
                                        1] = self.species[element]['theta'][-1]
            self.species[element]['AR'][-1, -2] = 0
        elif self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['AL'][-1, -
                                        1] = self.species[element]['theta'][-1] + s[-1]
            self.species[element]['AL'][-1, -2] = -s[-1]
            self.species[element]['AR'][-1, -
                                        1] = self.species[element]['theta'][-1] - s[-1]
            self.species[element]['AR'][-1, -2] = s[-1]
        # NOTE: Robin BC is not implemented yet
        elif self.species[element]['bc_bot_type'] in ['robin']:
            self.species[element]['AL'][-1, -
                                        1] = self.species[element]['theta'][-1]
            self.species[element]['AL'][-1, -2] = 0
            self.species[element]['AR'][-1, -
                                        1] = self.species[element]['theta'][-1]
            self.species[element]['AR'][-1, -2] = 0
            self.species[element]['AL'][-2, -
                                        2] = self.species[element]['theta'][-2] + s[-2]
            self.species[element]['AL'][-2, -3] = -s[-2] / 2
            self.species[element]['AL'][-2, -1] = -s[-2] / 2
            self.species[element]['AR'][-2, -
                                        2] = self.species[element]['theta'] - s
            self.species[element]['AR'][-2, -3] = s[-2] / 2
            self.species[element]['AR'][-2, -1] = s[-2] / 2
        else:
            print('\nABORT!!!: Not correct bottom boundary condition type...')
            sys.exit()

    def update_matrices_due_to_bc(self, element, i):
        s = self.species[element][
            'theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
        q = self.species[element]['theta'] * \
            self.w * self.dt / self.dx

        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])

        elif self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * self.species[element]['bc_bot'] * \
                (2 * s[-1] - q[-1]) * self.dx / \
                self.species[element]['theta'][-1] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * \
                (2 * s[0] - q[0]) * self.dx / \
                self.species[element]['theta'][0] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * \
                (2 * s[0] - q[0]) * self.dx / \
                self.species[element]['theta'][0] / self.species[element]['D']
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * self.species[element]['bc_bot'] * \
                (2 * s[-1] - q[-1]) * self.dx / \
                self.species[element]['theta'][-1] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['robin']:
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * \
                (2 * s[0] - q[0]) * self.dx / \
                self.species[element]['theta'][0] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['robin']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[
                element]['AR'].dot(self.profiles[element])

        else:
            print('\nABORT!!!: Not correct boundary condition...')
            sys.exit()

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
                    sys.exit()

    def integrate_one_timestep(self, i):
        self.transport_integrate(i)
        self.reactions_integrate(i)

    def reactions_integrate(self, i):
        C_new, rates = OdeSolver.ode_integrate(
            self.profiles, self.dcdt, self.rates, self.constants, self.dt, solver='rk4')

        for element in C_new:
            if element is not 'Temperature':
                # the concentration should be positive
                C_new[element][C_new[element] < 0] = 0
            self.profiles[element] = C_new[element]
            self.update_matrices_due_to_bc(element, i)
            self.species[element]['concentration'][:, i] = self.profiles[element]
            self.species[element]['rates'][:, i] = rates[element] / self.dt

    def transport_integrate(self, i):
        """ The possible place to parallel execution
        """
        if False:
            species = [e for e in self.species]
            self.parallel(delayed(self.transport_integrate_one_element)(
                e, i) for e in species)
        else:
            for element in self.species:
                self.transport_integrate_one_element(element, i)

    def transport_integrate_one_element(self, element, i):
        self.profiles[element] = OdeSolver.linear_alg_solver(self.species[element]['AL'], self.species[element]['B'])
        self.update_matrices_due_to_bc(element, i)
        self.species[element]['concentration'][:, i] = self.profiles[element]

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
        phi = self.phi

        if order == 4:
            flux = D * (-25 * phi[1] * C[1, idx] + 48 * phi[2] * C[2, idx] - 36 * phi[3] * C[
                        3, idx] + 16 * phi[4] * C[4, idx] - 3 * phi[5] * C[5, idx]) / self.dx / 12
        if order == 3:
            flux = D * (-11 * phi[1] * C[1, idx] + 18 * phi[2] * C[2, idx] -
                        9 * phi[3] * C[3, idx] + 2 * phi[4] * C[4, idx]) / self.dx / 6
        if order == 2:
            flux = D * (-3 * phi[1] * C[1, idx] + 4 * phi[2] *
                        C[2, idx] - phi[3] * C[3, idx]) / self.dx / 2
        if order == 1:
            flux = - D * (phi[0] * C[0, idx] - phi[2] * C[2, idx]) / 2 / self.dx

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
        phi = self.phi

        if order == 4:
            flux = D * (-25 * phi[-2] * C[-2, idx] + 48 * phi[-3] * C[-3, idx] - 36 * phi[-4] *
                        C[-4, idx] + 16 * phi[-5] * C[-5, idx] - 3 * phi[-6] * C[-6, idx]) / self.dx / 12
        if order == 3:
            flux = D * (-11 * phi[-2] * C[-2, idx] + 18 * phi[-3] * C[-3, idx] -
                        9 * phi[-4] * C[-4, idx] + 2 * phi[-5] * C[-5, idx]) / self.dx / 6
        if order == 2:
            flux = D * (-3 * phi[-2] * C[-2, idx] + 4 * phi[-3] *
                        C[-3, idx] - phi[-4] * C[-4, idx]) / self.dx / 2
        if order == 1:
            flux = - D * phi[-1](* C[-1, idx] - phi[-3] * C[-3, :]) / 2 / self.dx

        return flux

    def is_solute(self, element):
        return self.species[element]['is_solute']

    """Mapping of ploting methods from Ploter module"""

    custom_plot = Ploter.custom_plot
    plot_depths = Ploter.plot_depths
    plot_times = Ploter.plot_times
    plot_profiles = Ploter.plot_profiles
    plot_profile = Ploter.plot_profile
    plot_contourplots = Ploter.plot_contourplots
    contour_plot = Ploter.contour_plot
    plot_contourplots_of_rates = Ploter.plot_contourplots_of_rates
    contour_plot_of_rates = Ploter.contour_plot_of_rates
