import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy import special
import matplotlib.pyplot as plt
import numexpr as ne
import time
import sys
from collections import ChainMap
from joblib import Parallel, delayed


def paral_num_rates(rate, coef, conc, element):
    return {element: ne.evaluate(rate, {**coef, **conc})}


def ode_integrate(C0, dcdt, rates, coef, dt, parallel):
    """Integrates the reactions according to 4th Order Runge-Kutta method or Butcher 5th"""

    def k_loop(conc):
        rates_num = {}
        # rates_num = parallel(delayed(paral_num_rates)(v, coef, C0, e) for e, v in rates.items())
        # rates_num = dict(ChainMap(*rates_num))
        for element, rate in rates.items():
            rates_num[element] = ne.evaluate(rate, {**coef, **conc})

        Kn = {}
        for element in dcdt:
            Kn[element] = dt * ne.evaluate(dcdt[element], {**coef, **rates_num})

        return Kn

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

    return rk4(C0)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PorousMediaLab:
    """PorousMediaLab module solves Advection-Diffusion-Reaction Equation"""

    def __init__(self, length, dx, tend, dt, phi, w=0):
        # ne.set_num_threads(ne.detect_number_of_cores())
        self.length = length
        self.dx = dx
        self.tend = tend
        self.dt = dt
        self.phi = phi
        self.w = w
        self.species = DotDict({})
        self.profiles = DotDict({})
        self.dcdt = DotDict({})
        self.rates = DotDict({})
        self.estimated_rates = DotDict({})
        self.constants = DotDict({})
        self.x = np.linspace(0, length, length / dx + 1)
        self.time = np.linspace(0, tend, tend / dt + 1)
        self.N = self.x.size
        self.num_adjustments = 0
        self.parallel = Parallel(n_jobs=1)

    def __getattr__(self, attr):
        return self.species[attr]

    def add_temperature(self, D, init_temperature):
        self.species['Temperature'] = DotDict({})
        self.species['Temperature']['is_solute'] = True
        self.species['Temperature']['bc_top'] = init_temperature
        self.species['Temperature']['bc_top_type'] = 'dirichlet'
        self.species['Temperature']['bc_bot'] = init_temperature
        self.species['Temperature']['bc_bot_type'] = 'dirichlet'
        self.species['Temperature']['fx_bottom'] = 0
        self.species['Temperature']['theta'] = 1
        self.species['Temperature']['D'] = D
        self.species['Temperature']['init_C'] = init_temperature
        self.species['Temperature']['concentration'] = np.zeros((self.N, self.time.size))
        self.species['Temperature']['concentration'][:, 0] = (init_temperature * np.ones((self.N)))
        self.profiles['Temperature'] = self.species['Temperature']['concentration'][:, 0]
        self.template_AL_AR('Temperature')
        self.update_matrices_due_to_bc('Temperature', 0)
        self.dcdt['Temperature'] = '0'

    def add_species(self, is_solute, element, D, init_C, bc_top, bc_top_type, bc_bot, bc_bot_type):
        self.species[element] = DotDict({})
        self.species[element]['is_solute'] = is_solute
        self.species[element]['bc_top'] = bc_top
        self.species[element]['bc_top_type'] = bc_top_type
        self.species[element]['bc_bot'] = bc_bot
        self.species[element]['bc_bot_type'] = bc_bot_type
        self.species[element]['fx_bottom'] = 0
        self.species[element]['theta'] = self.phi if is_solute else 1 - self.phi
        self.species[element]['D'] = D
        self.species[element]['init_C'] = init_C
        self.species[element]['concentration'] = np.zeros((self.N, self.time.size))
        self.species[element]['concentration'][:, 0] = (init_C * np.ones((self.N)))
        self.profiles[element] = self.species[element]['concentration'][:, 0]
        self.template_AL_AR(element)
        self.update_matrices_due_to_bc(element, 0)
        self.dcdt[element] = '0'

    def add_solute_species(self, element, D, init_C, bc_top):
        self.add_species(True, element, D, init_C, bc_top, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='neumann')

    def add_solid_species(self, element, D, init_C, bc_top):
        self.add_species(False, element, D, init_C, bc_top, bc_top_type='neumann', bc_bot=0, bc_bot_type='neumann')

    def template_AL_AR(self, element):
        e1 = np.ones((self.N, 1))
        s = self.species[element]['theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx  #
        q = self.species[element]['theta'] * self.w * self.dt / self.dx
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
        else:
            print('\nABORT!!!: Not correct bottom boundary condition type...')
            sys.exit()

    def new_boundary_condition(self, element, bc):
        self.species[element]['bc_top'] = bc

    def update_matrices_due_to_bc(self, element, i):
        if self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.species[element]['concentration'][0, i] = self.species[element]['bc_top']
            self.species[element]['concentration'][-1, i] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.species[element]['concentration'][:, i])

        elif self.species[element]['bc_top_type'] in ['dirichlet', 'constant'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['concentration'][0, i] = self.species[element]['bc_top']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.species[element]['concentration'][:, i])
            s = self.species[element]['theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * self.species[element]['bc_bot'] * s * self.dx / self.species[element]['theta'] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['dirichlet', 'constant']:
            self.species[element]['concentration'][-1, i] = self.species[element]['bc_bot']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.species[element]['concentration'][:, i])
            s = self.species[element]['theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * s * self.dx / self.species[element]['theta'] / self.species[element]['D']

        elif self.species[element]['bc_top_type'] in ['neumann', 'flux'] and self.species[element]['bc_bot_type'] in ['neumann', 'flux']:
            self.species[element]['B'] = self.species[element]['AR'].dot(self.species[element]['concentration'][:, i])
            s = self.species[element]['theta'] * self.species[element]['D'] * self.dt / self.dx / self.dx
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * s * self.dx / self.species[element]['theta'] / self.species[element]['D']
            self.species[element]['B'][-1] = self.species[element]['B'][-1] + 2 * self.species[element]['bc_bot'] * s * self.dx / self.species[element]['theta'] / self.species[element]['D']
        else:
            print('\nABORT!!!: Not correct top boundary condition type...')
            sys.exit()

    def estimate_time_of_computation(self, i):
        if i == 1:
            self.tstart = time.time()
            if self.num_adjustments < 1:
                print("Simulation started:\n\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if i == 100:
            total_t = len(self.time) * (time.time() - self.tstart) / 100
            m, s = divmod(total_t, 60)
            h, m = divmod(m, 60)
            print("\n\nEstimated time of the code execution:\n\t %dh:%02dm:%02ds" % (h, m, s))
            print("Will finish approx.:\n\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_t)))

    def solve(self, do_adjust=True):
        if self.num_adjustments < 1:
            print('Simulation of sediment core with following params:\n\ttend = %.1f years,\n\tdt = %.2e years,\n\tL = %.1f,\n\tdx = %.2e,\n\tw = %.2f' % (self.time[-1], self.dt, self.length, self.dx, self.w))
        with np.errstate(invalid='raise'):
            try:
                for i in np.arange(1, len(self.time)):
                    self.integrate_one_timestep(i)
                    self.estimate_time_of_computation(i)
            except FloatingPointError as inst:
                if do_adjust and self.num_adjustments < 6:
                    print('\nWarning: Numerical instability. adjusting dt...')
                    self.restart_solve()
                else:
                    print('\nABORT!!!: Numerical instability. Time step was adjusted 5 times with no luck. Please, adjust dt and dx manually...')
                    sys.exit()

    def redefine_concentration_matrices(self):
        for element in self.species:
            self.species[element]['concentration'] = np.zeros((self.N, self.time.size))
            self.species[element]['concentration'][:, 0] = (self.species[element]['init_C'] * np.ones((self.N)))
            self.profiles[element] = self.species[element]['concentration'][:, 0]
            self.update_matrices_due_to_bc(element, 0)

    def adjust_timestep(self):
        self.dt /= 2
        self.time = np.linspace(0, self.tend, self.tend / self.dt + 1)
        self.num_adjustments += 1
        print('Time step was reduced to\n\tdt = %.2e.' % (self.dt))

    def restart_solve(self):
        self.adjust_timestep()
        self.redefine_concentration_matrices()
        self.solve()

    def integrate_one_timestep(self, i):
        self.transport_integrate(i)
        self.reactions_integrate(i)

    def transport_integrate_one_element(self, element, i):
        self.species[element]['concentration'][:, i] = linalg.spsolve(self.species[element]['AL'], self.species[element]['B'], use_umfpack=True)
        self.update_matrices_due_to_bc(element, i)
        self.profiles[element] = self.species[element]['concentration'][:, i]

    def transport_integrate(self, i):
        for element in self.species:
            self.transport_integrate_one_element(element, i)

    def reactions_integrate(self, i):
        C_new, rates = ode_integrate(self.profiles, self.dcdt, self.rates, self.constants, self.dt, self.parallel)

        for element in C_new:
            if element is not 'Temperature':
                C_new[element][C_new[element] < 0] = 0  # the concentration should be positive
            self.species[element]['concentration'][:, i] = C_new[element]
            self.update_matrices_due_to_bc(element, i)
            self.profiles[element] = self.species[element]['concentration'][:, i]
            self.estimated_rates[element] = rates[element] / self.dt

    def is_solute(self, element):
        return self.species[element]['is_solute']

    def plot_depths(self, element, depths=[0, 1, 2, 3, 4], years_to_plot=10):
        plt.figure()
        plt.title('Bulk ' + element + ' concentration')
        if self.time[-1] > years_to_plot:
            num_of_elem = int(years_to_plot / self.dt)
        else:
            num_of_elem = len(self.time)
        for depth in depths:
            lbl = str(depth) + ' cm'
            plt.plot(self.time[-num_of_elem:], self.species[element]['theta'] * self.species[element]['concentration'][int(depth / self.dx)][-num_of_elem:], label=lbl)
        plt.legend()
        plt.show()

    def plot_profiles(self):
        for element in sorted(self.species):
            self.plot_profile(element)

    def plot_profile(self, element):
        plt.figure()
        plt.plot(self.species[element]['theta'] * self.profiles[element], -self.x, label=element)
        if element == 'Temperature':
            plt.title('Temperature profile')
            plt.xlabel('Temperature, C')
        else:
            plt.title('Bulk ' + element + ' concentration')
            plt.xlabel('mmol/L')
        plt.ylabel('Depth, cm')
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_3_profiles(self, elements_to_plot=['O2', 'OM1', 'OM2']):
        # plt.title('Profiles')
        fig, ax = plt.subplots()

        # Twin the x-axis twice to make independent y-axes.
        axes = [ax, ax.twinx(), ax.twinx()]

        # Make some space on the right side for the extra y-axis.
        fig.subplots_adjust(right=0.75)

        # Move the last y-axis spine over to the right by 20% of the width of the axes
        axes[-1].spines['right'].set_position(('axes', 1.2))

        # To make the border of the right-most axis visible, we need to turn the frame
        # on. This hides the other plots, however, so we need to turn its fill off.
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)
        colors = ('g', 'r', 'b')
        for element, ax, color in zip(elements_to_plot, axes, colors):
            ax.plot(self.x, self.species[element]['theta'] * self.profiles[element], label=element, color=color)
            ax.set_ylabel('%s [mmol/L]' % element, color=color)
            ax.tick_params(axis='y', colors=color)

        plt.show()

    def plot_contourplots(self):
        for element in sorted(self.species):
            self.contour_plot(element)

    def contour_plot(self, element):
        plt.figure()
        plt.title('Bulk ' + element + ' concentration')
        X, Y = np.meshgrid(self.time[1::100], -self.x)
        CS = plt.contourf(X, Y, self.species[element]['theta'] * self.species[element]['concentration'][:, 0:-1:100], 10, cmap=plt.cm.ocean, origin='lower')
        plt.clabel(CS, inline=1, fontsize=10, colors='w')
        plt.colorbar(CS)
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        # cbar.ax.set_ylabel('verbosity coefficient')
        plt.show()


def transport_equation_test():
    D = 40
    w = 0
    t = 2
    dx = 0.1
    L = 100
    phi = 1
    dt = 0.001

    C = PorousMediaLab(L, dx, t, dt, phi, w)
    C.add_solute_species('O2', D, 0.0, 1)
    C.dcdt.O2 = '0'
    C.solve()

    x = np.linspace(0, L, L / dx + 1)
    sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

    plt.figure()
    plt.plot(x, sol, 'r', label='Analytical solution')
    plt.plot(C.x, C.O2.concentration[:, -1], 'kx', label='Numerical')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    transport_equation_test()
