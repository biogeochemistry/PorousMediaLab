import numpy as np

import porousmedialab.desolver as desolver
import porousmedialab.phcalc as phcalc
import porousmedialab.plotter as plotter
from porousmedialab.dotdict import DotDict
from porousmedialab.lab import Lab


class Column(Lab):
    """Column module solves Advection-Diffusion-Reaction Equation
    in porous media"""

    def __init__(self, length, dx, tend, dt, w=0, ode_method='scipy'):
        """ initializing the domain of the column model

        Arguments:
            length {float} -- the length of the domain
            dx {float} -- mesh size
            tend {float} -- time of simulation
            dt {float} -- timestep in the model

        Keyword Arguments:
            w {float} -- default advective flux for all species (default: {0})
            ode_method {str} -- method to solve ode (default: {'rk4'})
        """
        # ne.set_num_threads(ne.detect_number_of_cores())
        super().__init__(tend, dt)
        self.x = np.linspace(0, length, int(length / dx) + 1)
        self.N = self.x.size
        self.length = length
        self.dx = dx
        self.w = w
        self.ode_method = ode_method

    def add_species(self,
                    theta,
                    name,
                    D,
                    init_conc,
                    bc_top_value,
                    bc_top_type,
                    bc_bot_value,
                    bc_bot_type,
                    w=False,
                    int_transport=True):
        """add chemical compund to the column model with boundary
        conditions

        Arguments:
            theta {numpy.array} -- porosity or 1 minus porosity
            name {str} -- name of the element
            D {float} -- total diffusion
            init_conc {float or numpy.array} -- initial concentration
            bc_top_value {float} -- top boundary value
            bc_top_type {str} -- boundary type (flux, constant)
            bc_bot_value {float} -- bottom boundary value
            bc_bot_type {str} -- type of bottom boundary

        Keyword Arguments:
            w {float} -- advective term for this element (default: {False})
            int_transport {bool} -- integrate transport? (default: {True})
        """
        self.species[name] = DotDict({})
        self.species[name]['bc_top_value'] = bc_top_value
        self.species[name]['bc_top_type'] = bc_top_type.lower()
        self.species[name]['bc_bot_value'] = bc_bot_value
        self.species[name]['bc_bot_type'] = bc_bot_type.lower()
        self.species[name]['theta'] = np.ones((self.N)) * theta
        self.species[name]['D'] = D
        self.species[name]['init_conc'] = init_conc
        self.species[name]['concentration'] = np.zeros((self.N,
                                                           self.time.size))
        self.species[name]['rates'] = np.zeros((self.N, self.time.size))
        self.species[name]['concentration'][:, 0] = self.species[name][
            'init_conc']
        self.profiles[name] = self.species[name]['concentration'][:, 0]
        if w:
            self.species[name]['w'] = w
        else:
            self.species[name]['w'] = self.w
        self.species[name]['int_transport'] = int_transport
        if int_transport:
            self.template_AL_AR(name)
            self.update_matrices_due_to_bc(name, 0)
        self.dcdt[name] = '0'

    def save_final_profiles(self):
        """Saves init conditons from profiles of all species in the
        current folder in CSV file with the name of species
        """
        for p in self.profiles:
            np.savetxt(p + '.csv', np.array([self.x, self.profiles[p]]).T, delimiter=',')

    def load_initial_conditions(self):
        """Loads init conditons from profiles of all species in the
        current folder in CSV file with the name of species
        """
        for elem in self.species:
            init_values = np.loadtxt(elem + '.csv', delimiter=',')
            init_conc_at_x = init_values[:, 1]
            x = init_values[:, 0]
            init_conc = np.interp(self.x, x, init_conc_at_x)
            self.species[elem]['init_conc'] = init_conc
            self.species[elem]['concentration'][:, 0] = self.species[elem]['init_conc']
            self.profiles[elem] = self.species[elem]['concentration'][:, 0]
            self.template_AL_AR(elem)
            self.update_matrices_due_to_bc(elem, 0)

    def change_boundary_conditions(self, element, i, bc_top_value, bc_top_type,
                                   bc_bot_value, bc_bot_type):
        """Methods checks if boundary conditions are changed and if yes
        generates new matrices for solving PDE
        """

        if (self.species[element].bc_top_type != bc_top_type.lower()
                or self.species[element].bc_top_value != bc_top_value
                or self.species[element].bc_bot_type != bc_bot_type.lower()
                or self.species[element].bc_bot_value != bc_bot_value):
            print("Boundary conditions changed for {} at time {}".format(
                element, self.time[i]))
            self.species[element].bc_top_type = bc_top_type.lower()
            self.species[element].bc_top_value = bc_top_value
            self.species[element].bc_bot_type = bc_bot_type.lower()
            self.species[element].bc_bot_value = bc_bot_value
            self.template_AL_AR(element)
            self.update_matrices_due_to_bc(element, i)

    def template_AL_AR(self, element):
        """creates the templates of matrices for linear algebra solutions

        Arguments:
            element {str} -- name of the element for which it creates AL,AR
        """
        self.species[element]['AL'], self.species[
            element]['AR'] = desolver.create_template_AL_AR(
                self.species[element]['theta'], self.species[element]['D'],
                self.species[element]['w'],
                self.species[element]['bc_top_type'],
                self.species[element]['bc_bot_type'], self.dt, self.dx, self.N)

    def update_matrices_due_to_bc(self, element, i):
        """updating the matrices due to boundary conditions

        Arguments:
            element {str} -- name of the element
            i {int} -- number of the step
        """
        self.profiles[element], self.species[
            element]['B'] = desolver.update_matrices_due_to_bc(
                self.species[element]['AR'], self.profiles[element],
                self.species[element]['theta'], self.species[element]['D'],
                self.species[element]['w'],
                self.species[element]['bc_top_type'],
                self.species[element]['bc_top_value'],
                self.species[element]['bc_bot_type'],
                self.species[element]['bc_bot_value'], self.dt, self.dx, self.N)

        self.species[element]['concentration'][:, i] = self.profiles[element]

    def add_time_variable(self):
        # for now we just added it in the batch system. Not sure if we
        # need it here.
        pass

    def create_acid_base_system(self):
        self.add_species(
            theta=True,
            name='pH',
            D=0,
            init_conc=7,
            bc_top_value=7,
            bc_top_type='dirichlet',
            bc_bot_value=7,
            bc_bot_type='dirichlet',
            w=False,
            int_transport=False)
        self.acid_base_system = phcalc.System(
            *[c['pH_object'] for c in self.acid_base_components])

    def acid_base_update_concentrations(self, i):
        for component in self.acid_base_components:
            init_conc = 0
            alphas = component['pH_object'].alpha(
                self.species['pH']['concentration'][:, i])
            for idx in range(len(component['species'])):
                init_conc += self.species[component['species'][idx]][
                    'concentration'][:, i]
            for idx in range(len(component['species'])):
                self.species[component['species'][idx]][
                    'concentration'][:, i] = init_conc * alphas[:, idx]
                self.profiles[component['species'][idx]] = self.species[
                    component['species'][idx]]['concentration'][:, i]

    def integrate_one_timestep(self, i):
        if i < 2:
            self.pre_run_methods()
        self.transport_integrate(i)
        if self.henry_law_equations:
            self.henry_equilibrium_integrate(i)
        if self.acid_base_components:
            self.acid_base_equilibrium_solve(i)
        if self.rates:
            if self.ode_method == 'scipy':
                self.reactions_integrate_vectorized(i)  # Use vectorized by default
            elif self.ode_method == 'scipy_sequential':
                self.reactions_integrate_scipy(i)  # Keep for backward compat
            else:
                self.reactions_integrate(i)

    def reactions_integrate(self, i):
        C_new, rates_per_elem, rates_per_rate = desolver.ode_integrate(
            self.profiles,
            self.dcdt,
            self.rates,
            self.constants,
            self.dt,
            solver=self.ode_method)

        try:
            for rate_name, rate in rates_per_rate.items():
                self.estimated_rates[rate_name][:, i - 1] = rates_per_rate[
                    rate_name]
        except (KeyError, AttributeError):
            pass

        for element in C_new:
            if element != 'Temperature':
                # the concentration should be positive
                C_new[element][C_new[element] < 0] = 0
            self.profiles[element] = C_new[element]
            self.species[element]['concentration'][:, i] = self.profiles[
                element]
            self.species[element]['rates'][:, i] = rates_per_elem[
                element] / self.dt
            if self.species[element]['int_transport']:
                self.update_matrices_due_to_bc(element, i)

    def transport_integrate(self, i):
        """ Integrates transport equations
        """
        for element in self.species:
            if self.species[element]['int_transport']:
                self.transport_integrate_one_element(element, i)

    def transport_integrate_one_element(self, element, i):
        self.profiles[element] = desolver.linear_alg_solver(
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
        theta = self.species[elem]['theta']

        if order == 4:
            flux = D * (-25 * theta[1] * C[1, idx] + 48 * theta[2] * C[2, idx] - 36 * theta[3] * C[
                3, idx] + 16 * theta[4] * C[4, idx] - 3 * theta[5] * C[5, idx]) / self.dx / 12 \
                - theta[0] * self.species[elem]['w'] * C[0]
        if order == 3:
            flux = D * (-11 * theta[1] * C[1, idx] + 18 * theta[2] * C[2, idx] -
                        9 * theta[3] * C[3, idx] + 2 * theta[4] * C[4, idx]) / self.dx / 6 \
                - theta[0] * self.species[elem]['w'] * C[0]
        if order == 2:
            flux = D * (-3 * theta[1] * C[1, idx] + 4 *
                        theta[2] * C[2, idx] - theta[3] * C[3, idx]) / self.dx / 2 \
                - theta[0] * self.species[elem]['w'] * C[0]
        if order == 1:
            flux = - D * (theta[0] * C[0, idx] - theta[2] * C[2, idx]) / 2 / self.dx \
                - theta[0] * self.species[elem]['w'] * C[0]

        return flux

    def estimate_flux_at_bottom(self,
                                elem,
                                idx=slice(None, None, None),
                                order=4):
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
        theta = self.species[elem]['theta']

        if order == 4:
            flux = D * (-25 * theta[-2] * C[-2, idx] + 48 * theta[-3] * C[-3, idx] - 36 *
                        theta[-4] * C[-4, idx] + 16 * theta[-5] * C[-5, idx] - 3 * theta[-6] * C[-6, idx]) / self.dx / 12 \
                + theta[-1] * self.species[elem]['w'] * C[0]
        if order == 3:
            flux = D * (-11 * theta[-2] * C[-2, idx] + 18 * theta[-3] * C[-3, idx] -
                        9 * theta[-4] * C[-4, idx] + 2 * theta[-5] * C[-5, idx]) / self.dx / 6 \
                + theta[-1] * self.species[elem]['w'] * C[0]
        if order == 2:
            flux = D * (-3 * theta[-2] * C[-2, idx] + 4 *
                        theta[-3] * C[-3, idx] - theta[-4] * C[-4, idx]) / self.dx / 2 \
                + theta[-1] * self.species[elem]['w'] * C[0]
        if order == 1:
            flux = - D * (theta[-1] * C[-1, idx] - theta[-3] * C[-3, idx]) / 2 / self.dx \
                + theta[-1] * self.species[elem]['w'] * C[0]

        return flux

    """Mapping of plotting methods from plotter module"""

    custom_plot = plotter.custom_plot
    plot_depths = plotter.plot_depths
    plot_times = plotter.plot_times
    plot_profiles = plotter.plot_profiles
    plot_profile = plotter.plot_profile
    plot_contourplots = plotter.plot_contourplots
    contour_plot = plotter.contour_plot
    plot_contourplots_of_rates = plotter.plot_contourplots_of_rates
    contour_plot_of_rates = plotter.contour_plot_of_rates
    plot_contourplots_of_deltas = plotter.plot_contourplots_of_deltas
    contour_plot_of_delta = plotter.contour_plot_of_delta
    plot_saturation_index = plotter.saturation_index_countour
