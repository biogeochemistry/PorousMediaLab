import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import math
import numexpr as ne
import time
import sys
from scipy import special
import seaborn as sns
from matplotlib.colors import ListedColormap
sns.set_style("whitegrid")


def ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4'):
    """Integrates the reactions according to 4th Order Runge-Kutta method or Butcher 5th
    where the variables, rates, coef are passed as dictionaries
    """

    def k_loop(conc, dt=dt):
        rates_num = {}
        for element, rate in rates.items():
            rates_num[element] = ne.evaluate(rate, {**coef, **conc})

        Kn = {}
        for element in dcdt:
            Kn[element] = dt * ne.evaluate(dcdt[element], {**coef, **rates_num})

        return Kn

    def implicit_solver(C_0):

        class Derivative:

            def __init__(self, f, h=1E-5):
                self.f = f
                self.h = float(h)

            def __call__(self, x):
                f, h = self.f, self.h
                return (f(x + h) - f(x - h)) / (2 * h)

        def Newton(f, x, dfdx, epsilon=1.0E-7, N=100, store=False):
            f_value = f(x)
            n = 0
            if store:
                info = [(x, f_value)]
            while abs(f_value) > epsilon and n <= N:
                dfdx_value = float(dfdx(x))
                if abs(dfdx_value) < 1E-14:
                    raise ValueError("Newton: f’(%g)=%g" % (x, dfdx_value))

                x = x - f_value / dfdx_value
                n += 1
                f_value = f(x)
                if store:
                    info.append((x, f_value))
            if store:
                return x, info
            else:
                return x, n, f_value

        def F(w):
            return w - k_loop(w) - C0

        dFdw = Derivative(F)

        C_new = {}
        k1 = k_loop(C_0)
        for element in C_0:
            w_start = C_0[element] + k1[element]
            C_new[element], _ = Newton(F, w_start, dFdw, N=30)

        raise NotImplemented

    def sum_k(A, B, b):
        C_new = {}
        for k in A:
            C_new[k] = A[k] + b * B[k] * dt
        return C_new

    def rk4(C_0):
        """Integrates the reactions according to 4th Order Runge-Kutta method
            k_1 = dt*dcdt(C0, dt)
            k_2 = dt*dcdt(C0+0.5*k_1, dt)
            k_3 = dt*dcdt(C0+0.5*k_2, dt)
            k_4 = dt*dcdt(C0+k_3, dt)
            C_new = C0 + (k_1+2*k_2+2*k_3+k_4)/6
        """
        k1 = k_loop(C_0)
        k2 = k_loop(sum_k(C_0, k1, 0.5))
        k3 = k_loop(sum_k(C_0, k2, 0.5))
        k4 = k_loop(sum_k(C_0, k3, 1))
        C_new = {}
        num_rates = {}
        for element in C_0:
            num_rates[element] = (k1[element] + 2 * k2[element] + 2 * k3[element] + k4[element]) / 6
            C_new[element] = C_0[element] + num_rates[element]
        return C_new, num_rates

    def butcher5(C_0):
        """
        k_1 = dt*sediment_rates(C0, dt);
        k_2 = dt*sediment_rates(C0 + 1/4*k_1, dt);
        k_3 = dt*sediment_rates(C0 + 1/8*k_1 + 1/8*k_2, dt);
        k_4 = dt*sediment_rates(C0 - 1/2*k_2 + k_3, dt);
        k_5 = dt*sediment_rates(C0 + 3/16*k_1 + 9/16*k_4, dt);
        k_6 = dt*sediment_rates(C0 - 3/7*k_1 + 2/7*k_2 + 12/7*k_3 - 12/7*k_4 + 8/7*k_5, dt);
        C_new = C0 + (7*k_1 + 32*k_3 + 12*k_4 + 32*k_5 + 7*k_6)/90;
        """
        k1 = k_loop(C_0)
        k2 = k_loop(sum_k(C_0, k1, 1 / 4))
        k3 = k_loop(sum_k(sum_k(C_0, k1, 1 / 8), k2, 1 / 8))
        k4 = k_loop(sum_k(sum_k(C_0, k2, -0.5), k3, 1))
        k5 = k_loop(sum_k(sum_k(C_0, k1, 3 / 16), k4, 9 / 16))
        k6 = k_loop(sum_k(sum_k(sum_k(sum_k(sum_k(C_0, k1, -3 / 7), k2, 2 / 7), k3, 12 / 7), k4, -12 / 7), k5, 8 / 7))
        C_new = {}
        num_rates = {}
        for element in C_0:
            num_rates[element] = (7 * k1[element] + 32 * k3[element] + 12 * k4[element] + 32 * k5[element] + 7 * k6[element]) / 90
            C_new[element] = C_0[element] + num_rates[element]
        return C_new, num_rates

    if solver == 'butcher5':
        return butcher5(C0)
    if solver == 'implicit':
        return implicit_solver(C0)

    return rk4(C0)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PorousMediaLab:
    """PorousMediaLab module solves Advection-Diffusion-Reaction Equation in porous media"""

    def __init__(self, length, dx, tend, dt, phi, w=0):
        # ne.set_num_threads(ne.detect_number_of_cores())
        self.length = length
        self.dx = dx
        self.tend = tend
        self.dt = dt
        self.adjusted_dt = dt
        self.phi = phi
        self.w = w
        self.species = DotDict({})
        self.profiles = DotDict({})
        self.dcdt = DotDict({})
        self.rates = DotDict({})
        self.estimated_rates = DotDict({})
        self.constants = DotDict({})
        self.x = np.linspace(0, length, length / dx + 1)
        self.time = np.linspace(0, tend, round(tend / dt) + 1)
        self.N = self.x.size
        self.num_adjustments = 0

    def __getattr__(self, attr):
        return self.species[attr]

    def add_temperature(self, init_temperature, D=281000):
        self.species['Temperature'] = DotDict({})
        self.species['Temperature']['is_solute'] = True
        self.species['Temperature']['bc_top'] = init_temperature
        self.species['Temperature']['bc_top_type'] = 'dirichlet'
        self.species['Temperature']['bc_bot'] = 0
        self.species['Temperature']['bc_bot_type'] = 'neumann'
        self.species['Temperature']['theta'] = 1
        self.species['Temperature']['D'] = D
        self.species['Temperature']['init_C'] = init_temperature
        self.species['Temperature']['concentration'] = np.zeros((self.N, self.time.size))
        self.species['Temperature']['rates'] = np.zeros((self.N, self.time.size))
        self.species['Temperature']['concentration'][:, 0] = (init_temperature * np.ones((self.N)))
        self.profiles['Temperature'] = self.species['Temperature']['concentration'][:, 0]
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
        self.species[element]['theta'] = self.phi if is_solute else 1 - self.phi
        self.species[element]['D'] = D
        self.species[element]['init_C'] = init_C
        self.species[element]['concentration'] = np.zeros((self.N, self.time.size))
        self.species[element]['rates'] = np.zeros((self.N, self.time.size))
        self.species[element]['concentration'][:, 0] = self.species[element]['init_C']
        self.profiles[element] = self.species[element]['concentration'][:, 0]
        self.template_AL_AR(element)
        self.update_matrices_due_to_bc(element, 0)
        self.dcdt[element] = '0'

    def add_solute_species(self, element, D, init_C):
        self.add_species(True, element, D, init_C, bc_top=0, bc_top_type='neumann', bc_bot=0, bc_bot_type='neumann')

    def add_solid_species(self, element, init_C):
        self.add_species(False, element, 1e-18, init_C, bc_top=0, bc_top_type='neumann', bc_bot=0, bc_bot_type='neumann')

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
        e1 = np.ones((self.N, 1))
        s = self.species[element]['theta'] * self.species[element]['D'] * self.adjusted_dt / self.dx / self.dx  #
        q = self.species[element]['theta'] * self.w * self.adjusted_dt / self.dx
        self.species[element]['AL'] = spdiags(np.concatenate((e1 * (-s / 2 - q / 4), e1 * (self.species[element]['theta'] + s), e1 * (-s / 2 + q / 4)),
                                                             axis=1).T, [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[element]['AR'] = spdiags(np.concatenate((e1 * (s / 2 + q / 4), e1 * (self.species[element]['theta'] - s), e1 * (s / 2 - q / 4)),
                                                             axis=1).T, [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()

        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant']:
            self.species[element]['AL'][0, 0] = self.species[element]['theta']
            self.species[element]['AL'][0, 1] = 0
            self.species[element]['AR'][0, 0] = self.species[element]['theta']
            self.species[element]['AR'][0, 1] = 0
        elif self.species[element]['bc_top_type'] in ['neumann', 'flux']:
            self.species[element]['AL'][0, 0] = self.species[element]['theta'] + s
            self.species[element]['AL'][0, 1] = -s
            self.species[element]['AR'][0, 0] = self.species[element]['theta'] - s
            self.species[element]['AR'][0, 1] = s
        else:
            print('\nABORT!!!: Not correct top boundary condition type...')
            sys.exit()

        if self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.species[element]['AL'][-1, -1] = self.species[element]['theta']
            self.species[element]['AL'][-1, -2] = 0
            self.species[element]['AR'][-1, -1] = self.species[element]['theta']
            self.species[element]['AR'][-1, -2] = 0
        elif self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['AL'][-1, -1] = self.species[element]['theta'] + s
            self.species[element]['AL'][-1, -2] = -s
            self.species[element]['AR'][-1, -1] = self.species[element]['theta'] - s
            self.species[element]['AR'][-1, -2] = s
        # NOTE: Robin BC is not implemented yet
        elif self.species[element]['bc_bot_type'] in ['robin']:
            self.species[element]['AL'][-1, -1] = self.species[element]['theta']
            self.species[element]['AL'][-1, -2] = 0
            self.species[element]['AR'][-1, -1] = self.species[element]['theta']
            self.species[element]['AR'][-1, -2] = 0
            self.species[element]['AL'][-2, -2] = self.species[element]['theta'] + s
            self.species[element]['AL'][-2, -3] = -s / 2
            self.species[element]['AL'][-2, -1] = -s / 2
            self.species[element]['AR'][-2, -2] = self.species[element]['theta'] - s
            self.species[element]['AR'][-2, -3] = s / 2
            self.species[element]['AR'][-2, -1] = s / 2
        else:
            print('\nABORT!!!: Not correct bottom boundary condition type...')
            sys.exit()

    def update_matrices_due_to_bc(self, element, i):
        s = self.species[element]['theta'] * self.species[element]['D'] * self.adjusted_dt / self.dx / self.dx
        q = self.species[element]['theta'] * self.w * self.adjusted_dt / self.dx

        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])

        elif self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * self.species[element]['bc_bot'] * \
                (2 * s - q) * self.dx / self.species[element]['theta'] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * \
                (2 * s - q) * self.dx / self.species[element]['theta'] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * \
                (2 * s - q) * self.dx / self.species[element]['theta'] / self.species[element]['D']
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * self.species[element]['bc_bot'] * \
                (2 * s - q) * self.dx / self.species[element]['theta'] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['robin']:
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * \
                (2 * s - q) * self.dx / self.species[element]['theta'] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['robin']:
            self.profiles[element][0] = self.species[element]['bc_top']
            self.profiles[element][-1] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.profiles[element])

        else:
            print('\nABORT!!!: Not correct boundary condition...')
            sys.exit()

    def estimate_time_of_computation(self, i):
        if i == 1:
            self.tstart = time.time()
            if self.num_adjustments < 1:
                print("Simulation started:\n\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if i == 100:
            total_t = len(self.time) * (time.time() - self.tstart) / 100 * self.dt / self.adjusted_dt
            m, s = divmod(total_t, 60)
            h, m = divmod(m, 60)
            print("\n\nEstimated time of the code execution:\n\t %dh:%02dm:%02ds" % (h, m, s))
            print("Will finish approx.:\n\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_t)))

    def solve(self, do_adjust=True):
        if self.num_adjustments < 1:
            print('Simulation starts  with following params:\n\ttend = %.1f years,\n\tdt = %.2e years,\n\tL = %.1f,\n\tdx = %.2e,\n\tw = %.2f' %
                  (self.time[-1], self.adjusted_dt, self.length, self.dx, self.w))
        with np.errstate(invalid='raise'):
            try:
                for i in np.arange(1, len(np.linspace(0, self.tend, round(self.tend / self.adjusted_dt) + 1))):
                    self.integrate_one_timestep(i)
                    self.estimate_time_of_computation(i)
            except FloatingPointError as inst:
                if do_adjust and self.num_adjustments < 6:
                    print('\nWarning: Numerical instability. Adjusting dt...')
                    self.restart_solve()
                else:
                    print('\nABORT!!!: Numerical instability. Time step was adjusted 5 times with no luck. Please, adjust dt and dx manually...')
                    sys.exit()

    def reset_concentration_matrices(self):
        for element in self.species:
            self.species[element]['concentration'] = np.zeros((self.N, self.time.size))
            self.species[element]['concentration'][:, 0] = (self.species[element]['init_C'] * np.ones((self.N)))
            self.species[element]['rates'] = np.zeros((self.N, self.time.size))
            self.profiles[element] = self.species[element]['concentration'][:, 0]
            self.template_AL_AR(element)
            self.update_matrices_due_to_bc(element, 0)

    def adjust_timestep(self):
        self.adjusted_dt /= 10
        self.num_adjustments += 1
        print('Time step was reduced to\n\tdt = %.2e.' % (self.adjusted_dt))

    def restart_solve(self):
        self.adjust_timestep()
        self.reset_concentration_matrices()
        self.solve()

    def integrate_one_timestep(self, i):
        self.transport_integrate(i)
        self.reactions_integrate(i)

    def transport_integrate(self, i):
        for element in self.species:
            self.profiles[element] = linalg.spsolve(self.species[element]['AL'], self.species[element]['B'], use_umfpack=True)
            self.update_matrices_due_to_bc(element, i)
            if self.num_adjustments > 0:
                j = int(round(i * self.adjusted_dt / self.dt))
            else:
                j = i
            self.species[element]['concentration'][:, j] = self.profiles[element]

    def estimate_flux_at_top(self, elem, order=4):
        # % fourth-order
        # flux = D * (-25 * C(2) + 48 * C(3) - 36 * C(4) + 16 * C(5) - 3 * C(6)) / dx / 12;  %  [umol/cm^2/y]
        # % third order
        # % flux = D * (-11 * C(2) + 18 * C(3) - 9 * C(4) + 2 * C(5)) / dx / 6;  %  [umol/cm^2/y]
        # % flux = 0;  %  [umol/cm^2/y]
        # % second order
        # % flux = D * (-3 * C(2) + 4 * C(3) - C(4)) / dx / 2;  %  [umol/cm^2/y]
        # % first order
        # % flux = - D * (C(1) - C(3)) / 2 / dx;  %  [umol/cm^2/y]

        C = self.species[elem]['concentration']
        D = self.species[elem]['D']

        if order == 4:
            flux = D * (-25 * C[1, :] + 48 * C[2, :] - 36 * C[3, :] + 16 * C[4, :] - 3 * C[5, :]) / self.dx / 12
        if order == 3:
            flux = D * (-11 * C[1, :] + 18 * C[2, :] - 9 * C[3, :] + 2 * C[4, :]) / self.dx / 6
        if order == 2:
            flux = D * (-3 * C[1, :] + 4 * C[2, :] - C[3, :]) / self.dx / 2
        if order == 1:
            flux = - D * (C[0, :] - C[2, :]) / 2 / self.dx

        return flux

    def estimate_flux_at_bottom(self, elem, order=4):
        # % fourth-order
        # flux = D * (-25 * C(2) + 48 * C(3) - 36 * C(4) + 16 * C(5) - 3 * C(6)) / dx / 12;  %  [umol/cm^2/y]
        # % third order
        # % flux = D * (-11 * C(2) + 18 * C(3) - 9 * C(4) + 2 * C(5)) / dx / 6;  %  [umol/cm^2/y]
        # % flux = 0;  %  [umol/cm^2/y]
        # % second order
        # % flux = D * (-3 * C(2) + 4 * C(3) - C(4)) / dx / 2;  %  [umol/cm^2/y]
        # % first order
        # % flux = - D * (C(1) - C(3)) / 2 / dx;  %  [umol/cm^2/y]

        C = self.species[elem]['concentration']
        D = self.species[elem]['D']

        if order == 4:
            flux = D * (-25 * C[-2, :] + 48 * C[-3, :] - 36 * C[-4, :] + 16 * C[-5, :] - 3 * C[-6, :]) / self.dx / 12
        if order == 3:
            flux = D * (-11 * C[-2, :] + 18 * C[-3, :] - 9 * C[-4, :] + 2 * C[-5, :]) / self.dx / 6
        if order == 2:
            flux = D * (-3 * C[-2, :] + 4 * C[-3, :] - C[-4, :]) / self.dx / 2
        if order == 1:
            flux = - D * (C[-1, :] - C[-3, :]) / 2 / self.dx

        return flux

    def reactions_integrate(self, i):
        C_new, rates = ode_integrate(self.profiles, self.dcdt, self.rates, self.constants, self.adjusted_dt, solver='rk4')

        for element in C_new:
            if element is not 'Temperature':
                C_new[element][C_new[element] < 0] = 0  # the concentration should be positive
            self.profiles[element] = C_new[element]
            self.update_matrices_due_to_bc(element, i)
            if self.num_adjustments > 0:
                j = int(round(i * self.adjusted_dt / self.dt))
            else:
                j = i
            self.species[element]['concentration'][:, j] = self.profiles[element]
            self.species[element]['rates'][:, j] = rates[element] / self.adjusted_dt

    def is_solute(self, element):
        return self.species[element]['is_solute']

    def custom_plot(self, x, y, ttl='', y_lbl='', x_lbl=''):
        plt.figure()
        ax = plt.subplot(111)
        plt.plot(x, y, lw=3)
        plt.title(ttl)
        plt.xlim(x[0], x[-1])
        plt.ylabel(y_lbl)
        plt.xlabel(x_lbl)
        ax.grid(linestyle='-', linewidth=0.2)
        plt.show()

    def plot_depths(self, element, depths=[0, 1, 2, 3, 4], years_to_plot=10, days=True):
        plt.figure()
        ax = plt.subplot(111)
        if element == 'Temperature':
            plt.title('Temperature at specific depths')
            plt.ylabel('Temperature, C')
        else:
            plt.title(element + ' concentration at specific depths')
            plt.ylabel('mmol/L')
        if self.tend > years_to_plot:
            num_of_elem = int(years_to_plot / self.dt)
        else:
            num_of_elem = len(self.time)
        if days:
            t = self.time[-num_of_elem:] * 365
            plt.xlabel('Days, [day]')
        else:
            t = self.time[-num_of_elem:]
            plt.xlabel('Years, [year]')
        for depth in depths:
            lbl = str(depth) + ' cm'
            plt.plot(t, self.species[element]['concentration'][int(depth / self.dx)][-num_of_elem:], lw=3, label=lbl)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(linestyle='-', linewidth=0.2)
        plt.show()

    def plot_times(self, element, time_slices=[0, 1, 2, 3, 4]):
        plt.figure()
        ax = plt.subplot(111)
        if element == 'Temperature':
            plt.title('Temperature profile')
            plt.xlabel('Temperature, C')
        else:
            plt.title(element + ' concentration')
            plt.xlabel('mmol/L')
        plt.ylabel('Depth, cm')
        for tms in time_slices:
            lbl = '%.2f day' % (tms * 365)
            plt.plot(self.species[element]['concentration'][:, int(tms / self.dt)], -self.x, lw=3, label=lbl)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        ax.grid(linestyle='-', linewidth=0.2)
        plt.show()

    def plot_profiles(self):
        for element in sorted(self.species):
            self.plot_profile(element)

    def plot_profile(self, element):
        plt.figure()
        plt.plot(self.profiles[element], -self.x, sns.xkcd_rgb["denim blue"], lw=3, label=element)
        if element == 'Temperature':
            plt.title('Temperature profile after %.2f days' % (self.tend * 365))
            plt.xlabel('Temperature, C')
        else:
            plt.title('Bulk %s concentration after %.2f days' % (element, self.tend * 365))
            plt.xlabel('mmol/L')
        plt.ylabel('Depth, cm')
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        ax.grid(linestyle='-', linewidth=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_contourplots(self, **kwargs):
        for element in sorted(self.species):
            self.contour_plot(element, **kwargs)

    def contour_plot(self, element, labels=False, days=True, last_year=False):
        plt.figure()
        plt.title(element + ' concentration')
        resoluion = 100
        n = math.ceil(self.time.size / resoluion)
        if last_year:
            k = n - int(1 / self.dt)
        else:
            k = 1
        if days:
            X, Y = np.meshgrid(self.time[k::n] * 365, -self.x)
            plt.xlabel('Days, [day]')
        else:
            X, Y = np.meshgrid(self.time[k::n], -self.x)
            plt.xlabel('Years, [year]')
        z = self.species[element]['concentration'][:, k - 1:-1:n]
        CS = plt.contourf(X, Y, z, 51, cmap=ListedColormap(sns.color_palette("Blues", 51)), origin='lower')
        if labels:
            plt.clabel(CS, inline=1, fontsize=10, colors='w')
        cbar = plt.colorbar(CS)
        plt.ylabel('Depth, [cm]')
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        cbar.ax.set_ylabel('%s [mmol/L]' % element)
        if element == 'Temperature':
            plt.title('Temperature contour plot')
            cbar.ax.set_ylabel('Temperature, C')
        plt.show()

    def plot_contourplots_of_rates(self, **kwargs):
        elements = sorted(self.species)
        if 'Temperature' in elements:
            elements.remove('Temperature')
        for element in elements:
            self.contour_plot_of_rates(element, **kwargs)

    def contour_plot_of_rates(self, element, labels=False, days=True, last_year=False):
        plt.figure()
        plt.title('Rate of %s consumption/production' % element)
        resoluion = 100
        n = math.ceil(self.time.size / resoluion)
        if last_year:
            k = n - int(1 / self.dt)
        else:
            k = 1
        z = self.species[element]['rates'][:, k - 1:-1:n]
        lim = np.max(np.abs(z))
        lim = np.linspace(-lim - 0.1, +lim + 0.1, 51)
        if days:
            X, Y = np.meshgrid(self.time[k::n] * 365, -self.x)
            plt.xlabel('Days, [day]')
        else:
            X, Y = np.meshgrid(self.time[k::n], -self.x)
            plt.xlabel('Years, [year]')
        CS = plt.contourf(X, Y, z, 20, cmap=ListedColormap(sns.color_palette("RdBu_r", 101)), origin='lower', levels=lim, extend='both')
        if labels:
            plt.clabel(CS, inline=1, fontsize=10, colors='w')
        cbar = plt.colorbar(CS)
        plt.ylabel('Depth, [cm]')
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        cbar.ax.set_ylabel('Rate %s [mmol/L/yr]' % element)
        plt.show()


def transport_equation_plot():
    '''Check the transport equation integrator'''
    w = 5
    tend = 5
    dx = 0.1
    length = 100
    phi = 1
    dt = 0.001
    lab = PorousMediaLab(length, dx, tend, dt, phi, w)
    D = 5
    lab.add_solute_species('O2', D, 0.0, 1)
    lab.solve()
    x = np.linspace(0, lab.length, lab.length / lab.dx + 1)
    sol = 1 / 2 * (special.erfc((x - lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)) + np.exp(lab.w * x / D) * special.erfc((x + lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)))

    plt.figure()
    plt.plot(x, sol, 'k', label='Analytical solution')
    plt.scatter(lab.x[::10], lab.species['O2'].concentration[:, -1][::10], marker='x', label='Numerical')
    plt.xlim([x[0], x[-1]])
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    ax.grid(linestyle='-', linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def reaction_equation_plot():
    '''Check the reaction equation integrator'''
    C0 = {'C': 1}
    coef = {'k': 2}
    rates = {'R': 'k*C'}
    dcdt = {'C': '-R'}
    dt = 0.001
    T = 10
    time = np.linspace(0, T, T / dt + 1)
    num_sol = np.array(C0['C'])
    for i in range(1, len(time)):
        C_new, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')
        C0['C'] = C_new['C']
        num_sol = np.append(num_sol, C_new['C'])
    assert max(num_sol - np.exp(-coef['k'] * time)) < 1e-5

    plt.figure()
    plt.plot(time, np.exp(-coef['k'] * time), 'k', label='Analytical solution')
    plt.scatter(time[::100], num_sol[::100], marker='x', label='Numerical')
    plt.xlim([time[0], time[-1]])
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    ax.grid(linestyle='-', linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
