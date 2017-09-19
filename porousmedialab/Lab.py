import time
import sys
import traceback
import numpy as np
import numexpr as ne
from dotdict import DotDict
import phcalc
import desolver
import equilibriumsolver


class Lab:
    """The batch experiments simulations"""

    def __init__(self, tend, dt):
        self.tend = tend
        self.dt = dt
        self.time = np.linspace(0, tend, round(tend / dt) + 1)
        self.species = DotDict({})
        self.dynamic_functions = DotDict({})
        self.profiles = DotDict({})
        self.dcdt = DotDict({})
        self.rates = DotDict({})
        self.estimated_rates = DotDict({})
        self.constants = DotDict({})
        self.henry_law_equations = []
        self.acid_base_components = []
        self.acid_base_system = phcalc.System()

    def __getattr__(self, attr):
        return self.species[attr]

    def solve(self):
        with np.errstate(invalid='raise'):
            for i in np.arange(1, len(np.linspace(0, self.tend, round(self.tend / self.dt) + 1))):
                # try:
                self.integrate_one_timestep(i)
                self.estimate_time_of_computation(i)
                # except FloatingPointError as inst:
                #     print(
                #         '\nABORT!!!: Numerical instability... Please, adjust dt and dx manually...')
                #     traceback.print_exc()
                #     sys.exit()

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
            self.species[eq['gas']]['concentration'][:, i], self.species[eq['aq']]['concentration'][:, i] = equilibriumsolver.solve_henry_law(
                self.species[eq['aq']]['concentration'][:, i] + self.species[eq['gas']]['concentration'][:, i], eq['Hcc'])
            for elem in [eq['gas'], eq['aq']]:
                self.profiles[elem] = self.species[elem]['concentration'][:, i]
                if self.species[elem]['int_transport']:
                    self.update_matrices_due_to_bc(elem, i)

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

    def add_partition_equilibrium(self, aq, gas, Hcc):
        """ For partition reactions between 2 species

        Args:
            aq (string): name of aquatic species
            gas (string): name of gaseous species
            Hcc (double): Henry Law Constant
        """
        self.henry_law_equations.append({'aq': aq, 'gas': gas, 'Hcc': Hcc})

    def add_ion(self, element, charge):
        ion = phcalc.Neutral(charge=charge, conc=np.nan)
        self.acid_base_components.append(
            {'species': [element], 'pH_object': ion})

    def add_acid(self, species, pKa, charge=0):
        acid = phcalc.Acid(pKa=pKa, charge=charge, conc=np.nan)
        self.acid_base_components.append(
            {'species': species, 'pH_object': acid})

    def acid_base_equilibrium_solve(self, i):
        self.acid_base_solve_ph(i)
        self.acid_base_update_concentrations(i)

    def init_rates_arrays(self):
        for rate in self.rates:
            self.estimated_rates[rate] = np.zeros((self.N, self.time.size))

    def create_dynamic_functions(self):
        fun_str = desolver.create_ode_function(self.species, self.constants, self.rates, self.dcdt)
        exec(fun_str)
        self.dynamic_functions['dydt_str'] = fun_str
        self.dynamic_functions['dydt'] = locals()['f']
        self.dynamic_functions['solver'] = desolver.create_solver(locals()['f'])

    def pre_run_methods(self):
        if len(self.acid_base_components) > 0:
            self.create_acid_base_system()
            self.acid_base_equilibrium_solve(0)
        self.create_dynamic_functions()
        self.init_rates_arrays()

    def change_concentration_profile(self, element, i, new_profile):
        self.profiles[element] = new_profile
        self.update_matrices_due_to_bc(element, i)

    def reactions_integrate_scipy(self, i):
        # C_new, rates_per_elem, rates_per_rate = desolver.ode_integrate(self.profiles, self.dcdt, self.rates, self.constants, self.dt, solver='rk4')
        # C_new, rates_per_elem = desolver.ode_integrate(self.profiles, self.dcdt, self.rates, self.constants, self.dt, solver='rk4')
        # for idx_j in range(self.N):
        for idx_j in range(self.N):
            yinit = np.zeros(len(self.species))
            for idx, s in enumerate(self.species):
                yinit[idx] = self.profiles[s][idx_j]

            ynew = desolver.ode_integrate_scipy(self.dynamic_functions['solver'], yinit, self.dt)

            for idx, s in enumerate(self.species):
                self.species[s]['concentration'][idx_j, i] = ynew[idx]

        for element in self.species:
            self.profiles[element] = self.species[element]['concentration'][:, i]
            if self.species[element]['int_transport']:
                self.update_matrices_due_to_bc(element, i)

    def reconstruct_rates(self):
        for idx_t in range(len(self.time)):
            for name, rate in self.rates.items():
                conc = {}
                for s in self.species:
                    conc[s] = self.species[s]['concentration'][:, idx_t]
                r = ne.evaluate(rate, {**self.constants, **conc})
                self.estimated_rates[name][:, idx_t] = r * (r > 0)

        for s in self.species:
            self.species[s]['rates'] = (self.species[s]['concentration'][:, 1:] - self.species[s]['concentration'][:, :-1]) / self.dt
