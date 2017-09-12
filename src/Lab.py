import time
import sys
import traceback
import numpy as np
from DotDict import DotDict
import pHcalc
import DESolver
import EquilibriumSolver


class Lab:
    """The batch experiments simulations"""

    def __init__(self, tend, dt):
        self.tend = tend
        self.dt = dt
        self.time = np.linspace(0, tend, round(tend / dt) + 1)
        self.species = DotDict({})
        self.profiles = DotDict({})
        self.dcdt = DotDict({})
        self.rates = DotDict({})
        self.estimated_rates = DotDict({})
        self.constants = DotDict({})
        self.henry_law_equations = []
        self.acid_base_components = []
        self.acid_base_system = pHcalc.System()

    def __getattr__(self, attr):
        return self.species[attr]

    def solve(self):
        with np.errstate(invalid='raise'):
            for i in np.arange(1, len(np.linspace(0, self.tend, round(self.tend / self.dt) + 1))):
                try:
                    self.integrate_one_timestep(i)
                    self.estimate_time_of_computation(i)
                except FloatingPointError as inst:
                    print(
                        '\nABORT!!!: Numerical instability... Please, adjust dt and dx manually...')
                    traceback.print_exc()
                    sys.exit()

    def estimate_time_of_computation(self, i):
        if i == 1:
            self.tstart = time.time()
            print("Simulation started:\n\t", time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()))
        if i == 100:
            total_t = len(self.time) * (time.time() -
                                        self.tstart) / 100 * self.dt / self.dt
            m, s = divmod(total_t, 60)
            h, m = divmod(m, 60)
            print(
                "\n\nEstimated time of the code execution:\n\t %dh:%02dm:%02ds" % (h, m, s))
            print("Will finish approx.:\n\t", time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_t)))

    def henry_equilibrium_integrate(self, i):
        for eq in self.henry_law_equations:
            self.species[eq['gas']]['concentration'][:, i], self.species[eq['aq']]['concentration'][:, i] = EquilibriumSolver.solve_henry_law(
                self.species[eq['aq']]['concentration'][:, i] + self.species[eq['gas']]['concentration'][:, i], eq['Hcc'])
            for elem in [eq['gas'], eq['aq']]:
                self.profiles[elem] = self.species[elem]['concentration'][:, i]
                if self.species[elem]['int_transport']:
                    self.update_matrices_due_to_bc(elem, i)

    def reactions_integrate(self, i):
        C_new, rates_elem, rates_rate = DESolver.ode_integrate(
            self.profiles, self.dcdt, self.rates, self.constants, self.dt, solver='rk4')

        for rate_name, rate in rates_rate.items():
            self.estimated_rates[rate_name][:, i - 1] = rates_rate[rate_name]

        for element in C_new:
            if element is not 'Temperature':
                # the concentration should be positive
                C_new[element][C_new[element] < 0] = 0
            self.profiles[element] = C_new[element]
            self.species[element]['concentration'][:, i] = self.profiles[element]
            self.species[element]['rates'][:, i] = rates_elem[element] / self.dt
            if self.species[element]['int_transport']:
                self.update_matrices_due_to_bc(element, i)

    def acid_base_solve_ph(self, i):
        # initial guess from previous time-step
        res = self.species['pH']['concentration'][0, i - 1]
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
        self.acid_base_components.append(
            {'species': [element], 'pH_object': ion})

    def add_acid(self, species, pKa, charge=0):
        acid = pHcalc.Acid(pKa=pKa, charge=charge, conc=np.nan)
        self.acid_base_components.append(
            {'species': species, 'pH_object': acid})

    def acid_base_equilibrium_solve(self, i):
        self.acid_base_solve_ph(i)
        self.acid_base_update_concentrations(i)

    def init_rates_arrays(self):
        for rate in self.rates:
            self.estimated_rates[rate] = np.zeros((self.N, self.time.size - 1))

    def pre_run_methods(self):
        self.init_rates_arrays()
        if len(self.acid_base_components) > 0:
            self.create_acid_base_system()
            self.acid_base_equilibrium_solve(0)

    def change_concentration_profile(self, element, i, new_profile):
        self.profiles[element] = new_profile
        self.update_matrices_due_to_bc(element, i)
