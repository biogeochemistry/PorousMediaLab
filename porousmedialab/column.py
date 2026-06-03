import numpy as np
import numexpr as ne
from pathlib import Path

import porousmedialab.desolver as desolver
import porousmedialab.phcalc as phcalc
import porousmedialab.plotter as plotter
from porousmedialab.dotdict import DotDict
from porousmedialab.lab import Lab, build_regular_grid


# Finite-difference flux stencils keyed by derivative order. Each entry is
# (indices, coefficients, divisor) for the diffusive term. Top and bottom share
# the coefficients/divisor but mirror the index map; order 1 uses an irregular
# stencil that skips the first interior point.
_FLUX_STENCIL_TOP = {
    4: ((1, 2, 3, 4, 5), (-25, 48, -36, 16, -3), 12),
    3: ((1, 2, 3, 4), (-11, 18, -9, 2), 6),
    2: ((1, 2, 3), (-3, 4, -1), 2),
    1: ((0, 2), (-1, 1), 2),
}
_FLUX_STENCIL_BOTTOM = {
    4: ((-2, -3, -4, -5, -6), (-25, 48, -36, 16, -3), 12),
    3: ((-2, -3, -4, -5), (-11, 18, -9, 2), 6),
    2: ((-2, -3, -4), (-3, 4, -1), 2),
    1: ((-1, -3), (-1, 1), 2),
}


class Column(Lab):
    """Column module solves Advection-Diffusion-Reaction Equation
    in porous media"""

    def __init__(self, length, dx, tend, dt, w=0, ode_method='scipy',
                 numexpr_threads=None):
        """ initializing the domain of the column model

        Arguments:
            length {float} -- the length of the domain
            dx {float} -- mesh size
            tend {float} -- time of simulation
            dt {float} -- timestep in the model

        Keyword Arguments:
            w {float} -- default advective flux for all species (default: {0})
            ode_method {str} -- method to solve ode (default: {'scipy'})
            numexpr_threads {int} -- optional numexpr thread count
        """
        if numexpr_threads is not None:
            if numexpr_threads <= 0:
                raise ValueError(
                    f"numexpr_threads must be positive, got {numexpr_threads}"
                )
            ne.set_num_threads(numexpr_threads)
        super().__init__(tend, dt)
        self.x = build_regular_grid(0, length, dx, "space")
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
                    w=None,
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
            w {float} -- advective term for this element (default: column value)
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
        if w is None or w is False:
            self.species[name]['w'] = self.w
        else:
            self.species[name]['w'] = w
        self.species[name]['int_transport'] = int_transport
        if int_transport:
            self.template_AL_AR(name)
            self.update_matrices_due_to_bc(name, 0)
        self.dcdt[name] = '0'

    def save_final_profiles(self, directory='.'):
        """Saves init conditons from profiles of all species in the
        current folder in CSV file with the name of species
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for p in self.profiles:
            np.savetxt(
                directory / f'{p}.csv',
                np.array([self.x, self.profiles[p]]).T,
                delimiter=',')

    def load_initial_conditions(self, directory='.'):
        """Loads init conditons from profiles of all species in the
        current folder in CSV file with the name of species
        """
        directory = Path(directory)
        for elem in self.species:
            init_values = np.loadtxt(directory / f'{elem}.csv', delimiter=',')
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
            w=None,
            int_transport=False)
        self.acid_base_system = phcalc.System(
            *[c['pH_object'] for c in self.acid_base_components])

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

    def _estimate_flux(self, elem, idx, order, at_top):
        """Finite-difference flux estimator shared by the top and bottom BCs.

        ``estimate_flux_at_top`` and ``estimate_flux_at_bottom`` differ only in
        the stencil index map and the sign of the advective term, so both
        delegate here. The diffusive sum keeps the original scalar
        multiply / left-to-right addition order to avoid floating-point drift.
        """
        C = self.species[elem]['concentration']
        D = self.species[elem]['D']
        theta = self.species[elem]['theta']
        w = self.species[elem]['w']

        stencils = _FLUX_STENCIL_TOP if at_top else _FLUX_STENCIL_BOTTOM
        if order not in stencils:
            raise ValueError("order must be one of 1, 2, 3, or 4")
        indices, coeffs, divisor = stencils[order]
        bc_index, advective_sign = (0, -1) if at_top else (-1, 1)

        diffusive = D * sum(
            coeff * theta[k] * C[k, idx]
            for coeff, k in zip(coeffs, indices)) / self.dx / divisor
        advective = theta[bc_index] * w * C[bc_index, idx]
        return diffusive + advective_sign * advective

    def estimate_flux_at_top(self, elem, idx=slice(None, None, None), order=4):
        """
        Function estimates flux at the top BC

        Args:
            elem (TYPE): name of the element
            order (int, optional): order of the Derivative

        Returns:
            TYPE: estimated flux in time

        """
        return self._estimate_flux(elem, idx, order, at_top=True)

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
        return self._estimate_flux(elem, idx, order, at_top=False)

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
