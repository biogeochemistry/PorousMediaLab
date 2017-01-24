import numpy as np
import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy import special
import matplotlib.pyplot as plt
import numexpr as ne


class Sediment:
    """Sediment module solves Advection-Diffusion-Reaction Equation"""

    def __init__(self, length, dx, tend, dt, phi, w):
        self.length = length
        self.dx = dx
        self.dt = dt
        self.phi = phi
        self.w = w
        self.species = {}
        self.profiles = {}
        self.dcdt = {}
        self.rates = {}
        self.estimated_rates = {}
        self.constants = {}
        self.x = np.linspace(0, length, length / dx + 1)
        self.time = np.linspace(0, tend, tend / dt + 1)
        self.N = self.x.size

    def add_species(self, element, D, init_C, BC_top, is_solute):
        self.species[element] = {}
        self.species[element]['solute?'] = is_solute
        self.species[element]['bc_top'] = BC_top
        self.species[element]['fx_bottom'] = 0
        self.species[element]['D'] = D
        self.species[element]['concentration'] = np.zeros((self.N, self.time.size))
        self.species[element]['concentration'][:, 0] = (init_C * np.ones((self.N)))
        self.profiles[element] = self.species[element]['concentration'][:, 0]
        self.create_AL_AR(element)
        self.update_matrices_due_to_bc(element, 0)
        self.dcdt[element] = 0

    def add_solute_species(self, element, D, init_C, BC_top):
        self.add_species(element, D, init_C, BC_top, True)

    def add_solid_species(self, element, D, init_C, BC_top):
        self.add_species(element, D, init_C, BC_top, False)

    def template_AL_AR(self, element, theta):
        e1 = np.ones((self.N, 1))
        s = theta * self.species[element]['D'] * self.dt / self.dx / self.dx  #
        q = theta * self.w * self.dt / self.dx
        self.species[element]['AL'] = spdiags(np.concatenate((e1 * (-s / 2 - q / 4), e1 * (theta + s), e1 * (-s / 2 + q / 4)),
                                                             axis=1).T, [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[element]['AR'] = spdiags(np.concatenate((e1 * (s / 2 + q / 4), e1 * (theta - s), e1 * (s / 2 - q / 4)),
                                                             axis=1).T, [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[element]['AL'][-1, -1] = theta + s
        self.species[element]['AL'][-1, -2] = -s
        self.species[element]['AR'][-1, -1] = theta - s
        self.species[element]['AR'][-1, -2] = s

        if self.is_solute(element):
            self.species[element]['AL'][0, 0] = theta
            self.species[element]['AL'][0, 1] = 0
            self.species[element]['AR'][0, 0] = theta
            self.species[element]['AR'][0, 1] = 0
        else:
            self.species[element]['AL'][0, 0] = theta + s
            self.species[element]['AL'][0, 1] = -s
            self.species[element]['AR'][0, 0] = theta - s
            self.species[element]['AR'][0, 1] = s

    def create_AL_AR(self, element):
        # create_AL_AR: creates AL and AR matrices
        theta = self.phi if self.is_solute(element) else 1 - self.phi
        self.template_AL_AR(element, theta)

    def new_boundary_condition(self, element, bc):
        self.species[element]['bc_top'] = bc

    def update_matrices_due_to_bc(self, element, i):
        if self.is_solute(element):
            self.species[element]['concentration'][0, i] = self.species[element]['bc_top']
            self.species[element]['B'] = self.species[element]['AR'].dot(self.species[element]['concentration'][:, i])
        else:
            self.species[element]['B'] = self.species[element]['AR'].dot(self.species[element]['concentration'][:, i])
            s = (1 - self.phi) * self.species[element]['D'] * self.dt / self.dx / self.dx
            self.species[element]['B'][0] = self.species[element]['B'][0] + 2 * self.species[element]['bc_top'] * s * self.dx / (1 - self.phi) / self.species[element]['D']

    def solve(self):
        for i in np.arange(1, len(self.time)):
            self.integrate_one_timestep(i)

    def integrate_one_timestep(self, i):
        self.transport_integrate(i)
        self.reactions_integrate(i)

    def transport_integrate(self, i):
        for element in self.species:
            self.species[element]['concentration'][:, i] = linalg.spsolve(self.species[element]['AL'], self.species[element]['B'], use_umfpack=True)
            self.update_matrices_due_to_bc(element, i)
            self.profiles[element] = self.species[element]['concentration'][:, i]

    def reactions_integrate(self, i):
        C_new, rates = runge_kutta_integrate(self.profiles, self.dcdt, self.rates, self.constants, self.dt)

        for element in C_new:
            self.species[element]['concentration'][:, i] = C_new[element] * (C_new[element] > 0)
            self.update_matrices_due_to_bc(element, i)
            self.profiles[element] = self.species[element]['concentration'][:, i]
            self.estimated_rates[element] = rates[element]

    def is_solute(self, element):
        return self.species[element]['solute?']

    def plot_depths(self, element, depths=[0, 1, 2, 3, 4], years_to_plot=10):
        plt.figure()
        plt.title(element + ' concentrations')
        if self.time[-1] > years_to_plot:
            num_of_elem = int(years_to_plot / self.dt)
        else:
            num_of_elem = len(self.time)
        theta = self.phi if self.is_solute(element) else 1 - self.phi
        for depth in depths:
            lbl = str(depth) + ' cm'
            plt.plot(self.time[-num_of_elem:], theta * self.species[element]['concentration'][int(depth / self.dx)][-num_of_elem:], label=lbl)
        plt.legend()
        plt.show()

    def plot_profiles(self):
        plt.figure()
        plt.title('Profiles')
        for element in self.species:
            theta = self.phi if self.is_solute(element) else 1 - self.phi
            plt.plot(self.x, theta * self.profiles[element], label=element)
        plt.legend()
        plt.show()


def runge_kutta_integrate(C0, dcdt, rates, coef, dt):
    """Integrates the reactions according to 4th Order Runge-Kutta method
     k_1 = dt*dcdt(C0, dt)
     k_2 = dt*dcdt(C0+0.5*k_1, dt)
     k_3 = dt*dcdt(C0+0.5*k_2, dt)
     k_4 = dt*dcdt(C0+k_3, dt)
     C_new = C0 + (k_1+2*k_2+2*k_3+k_4)/6
    """

    def k_loop(conc):
        rates_num = {}

        for element in rates:
            rates_num[element] = ne.evaluate(rates[element], {**coef, **conc})

        dcdt_num = {}

        for element in dcdt:
            dcdt_num[element] = ne.evaluate(dcdt[element], rates_num)

        Kn = {}

        for element in conc:
            Kn[element] = dt * dcdt_num[element]
        return Kn

    def sum_k(A, B, b):
        C_new = {}
        for k in A:
            C_new[k] = A[k] + b * B[k] * dt
        return C_new

    def rk4(C_0):
        k1 = k_loop(C_0)
        k2 = k_loop(sum_k(C_0, k1, 0.5))
        k3 = k_loop(sum_k(C_0, k2, 0.5))
        k4 = k_loop(sum_k(C_0, k3, 1))
        C_new = {}
        num_rates = {}
        for element in C_0:
            num_rates[element] = (k1[element] + 2 * k2[element] + 2 * k3[element] + k4[element]) / 6 / dt
            C_new[element] = C_0[element] + num_rates[element] * dt
        return C_new, num_rates

    return rk4(C0)


if __name__ == '__main__':
    D = 269
    w = 1
    t = 50
    dx = 0.1
    L = 10
    phi = 0.9
    dt = 0.001
    rho = 2

    time = np.linspace(0, t, t / dt + 1)

    sediment = Sediment(L, dx, t, dt, phi, w)
    sediment.add_solute_species('O2', D, 0.0, 0.15)
    sediment.add_solid_species('OM', 0.1, 1., 0.1)
    sediment.constants['k'] = 0.1
    sediment.constants['Km_O2'] = 0.0123
    sediment.rates['R1'] = 'k * OM * O2/(O2+Km_O2)'
    sediment.dcdt['O2'] = '-4 * R1'
    sediment.dcdt['OM'] = '-R1'

    for i in np.arange(1, len(time)):
        bc = 0.15 + 0.1 * np.sin(time[i] * 2 * 3.14)
        sediment.new_boundary_condition('O2', bc)
        sediment.integrate_one_timestep(i)

    # sediment.solve()

    # x = np.linspace(0, L, L / dx + 1)
    # sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

    # Bulk concentrations
    # TODO how to convert bulk to mol/mg and conversion factor F?
    # plt.figure()
    # plt.title('Bulk concentrations')
    # plt.plot(sediment.x, phi * sediment.species['O2']['concentration'][:, -1], 'k')
    # plt.plot(sediment.x, (1 - phi) * sediment.species['OM']['concentration'][:, -1], 'b')
    # plt.show()

    # plt.figure()
    # plt.title('Aq and solid concentrations')
    # plt.plot(sediment.x, sediment.species['O2']['concentration'][:, -1], 'k')
    # # plt.plot(sediment.x, sediment.species['OM']['concentration'][:, -1], 'b')
    # plt.show()
