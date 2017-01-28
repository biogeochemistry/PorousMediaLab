import numpy as np
import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy import special
import matplotlib.pyplot as plt
import numexpr as ne
from scipy import special
from numba import jit, vectorize
import time
import warnings


# np.seterr(all='warn')
# warnings.filterwarnings('error')

def runge_kutta_integrate(C0, dcdt, rates, coef, dt):
    """Integrates the reactions according to 4th Order Runge-Kutta method or Butcher 5th"""

    def k_loop(conc):
        rates_num = {}
        for element in rates:
            rates_num[element] = ne.evaluate(rates[element], {**coef, **conc})

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

    return butcher5(C0)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Sediment:
    """Sediment module solves Advection-Diffusion-Reaction Equation"""

    def __init__(self, length, dx, tend, dt, phi, w):
        # ne.set_num_threads(ne.detect_number_of_cores())
        self.length = length
        self.dx = dx
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

    def __getattr__(self, attr):
        return self.species[attr]['concentration']

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
        print('Simulation of sediment core with following params:\n tend = %.1f years,\n dt = %.0eyears,\n L = %.1f,\n dx = %.0e,\n w = %.2f' % (self.time[-1], self.dt, self.length, self.dx, self.w))
        for i in np.arange(1, len(self.time)):
            if i == 1:
                tstart = time.time()
                print("Simulation started:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            self.integrate_one_timestep(i)
            if i == 100:
                total_t = len(self.time) * (time.time() - tstart) / 100
                m, s = divmod(total_t, 60)
                h, m = divmod(m, 60)
                print("Estimated time of the code execution: %dh:%02dm:%02ds" % (h, m, s))
                print("Will end approx.:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + total_t)))

    def integrate_one_timestep(self, i, do_adjust=True):
        with np.errstate(invalid='raise'):
            try:
                self.transport_integrate(i)
                self.reactions_integrate(i)
            except FloatingPointError:
                print('Warning: Numerical instability. Please, adjust dt and dx.')
                r =
                if do_adjust:

                    pass
                else:
                    raise


    def transport_integrate(self, i):
        for element in self.species:
            self.species[element]['concentration'][:, i] = linalg.spsolve(self.species[element]['AL'], self.species[element]['B'], use_umfpack=True)
            self.update_matrices_due_to_bc(element, i)
            self.profiles[element] = self.species[element]['concentration'][:, i]

    def reactions_integrate(self, i):
        C_new, rates = runge_kutta_integrate(self.profiles, self.dcdt, self.rates, self.constants, self.dt)

        for element in C_new:
            C_new[element][C_new[element] < 0] = 0  # the concentration should be positive
            self.species[element]['concentration'][:, i] = C_new[element]
            self.update_matrices_due_to_bc(element, i)
            self.profiles[element] = self.species[element]['concentration'][:, i]
            self.estimated_rates[element] = rates[element] / self.dt

    def is_solute(self, element):
        return self.species[element]['solute?']

    def plot_depths(self, element, depths=[0, 1, 2, 3, 4], years_to_plot=10):
        plt.figure()
        plt.title('Bulk ' + element + ' concentration')
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

    def plot_all_profiles(self):
        for element in self.species:
            self.plot_profile(element)

    def plot_profile(self, element):
        plt.figure()
        plt.title('Profiles')
        theta = self.phi if self.is_solute(element) else 1 - self.phi
        plt.plot(self.x, theta * self.profiles[element], label=element)
        plt.legend()
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
            theta = self.phi if self.is_solute(element) else 1 - self.phi
            ax.plot(self.x, theta * self.profiles[element], label=element, color=color)
            ax.set_ylabel('%s [mmol/L]' % element, color=color)
            ax.tick_params(axis='y', colors=color)

        plt.show()


def transport_equation_test():
    D = 40
    w = 0
    t = 2
    dx = 0.1
    L = 100
    phi = 1
    dt = 0.001

    C = Sediment(L, dx, t, dt, phi, w)
    C.add_solute_species('O2', D, 0.0, 1)
    C.dcdt.O2 = '0'
    C.solve()

    x = np.linspace(0, L, L / dx + 1)
    sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

    plt.figure()
    plt.plot(x, sol, 'r', label='Analytical solution')
    plt.plot(C.x, C.O2[:, -1], 'kx', label='Numerical')
    plt.legend()
    plt.show()


def lake_sediments():
    D = 368
    w = 0.2
    t = 30
    dx = 0.1
    L = 25
    phi = 0.9
    dt = 0.0001
    rho = 2
    Init_C = 0.231
    bc = 0.231

    time = np.linspace(0, t, t / dt + 1)

    sediment = Sediment(L, dx, t, dt, phi, w)
    sediment.add_solute_species('O2', D, Init_C, bc)
    sediment.add_solid_species('OM1', 5, 0, 1)
    sediment.add_solid_species('OM2', 5, 0, 1)
    sediment.add_solute_species('NO3', 359, 1.5e-3, 0)
    sediment.add_solid_species('FeOH3', 5, 0, 75)
    sediment.add_solute_species('SO4', 189, 28, 0)
    sediment.add_solute_species('NH4', 363, 22e-3, 0)
    sediment.add_solute_species('Fe2', 127, 0, 0)
    sediment.add_solid_species('FeOOH', 5, 0, 0)
    sediment.add_solute_species('H2S', 284, 0, 0)
    sediment.add_solute_species('HS', 284, 0, 0)
    sediment.add_solid_species('FeS', 5, 0, 0)
    sediment.add_solute_species('S0', 100, 0, 0)
    sediment.add_solute_species('PO4', 104, 0, 0)
    sediment.add_solid_species('S8', 5, 0, 0)
    sediment.add_solid_species('FeS2', 5, 0, 0)
    sediment.add_solid_species('AlOH3', 5, 0, 0)
    sediment.add_solid_species('PO4adsa', 5, 0, 0)
    sediment.add_solid_species('PO4adsb', 5, 0, 0)
    sediment.add_solute_species('Ca2', 141, 0, 0)
    sediment.add_solid_species('Ca3PO42', 5, 0, 0)
    sediment.add_solid_species('OMS', 5, 0, 0)

    sediment.constants = {'k_OM1': 1, 'k_OM2': 0.1, 'Km_O2': 0.02, 'Km_NO3': 0.005, 'Km_FeOH3': 50, 'Km_FeOOH': 50, 'Km_SO4': 1.6, 'Km_oxao': 0.001, 'Km_amao': 0.1, 'Kin_O2': 0.3292, 'Kin_NO3': 0.1, 'Kin_FeOH3': 0.1, 'Kin_FeOOH': 0.1, 'k_amox': 2000, 'k_Feox': 8.7e1, 'k_Sdis': 0.1, 'k_Spre': 2500, 'k_FeS2pre': 3.17, 'k_alum': 0.1,
                          'k_pdesorb_a': 1.35, 'k_pdesorb_b': 1.35, 'k_rhom': 6500, 'k_tS_Fe': 0.1, 'Ks_FeS': 2510, 'k_Fe_dis': 0.001, 'k_Fe_pre': 21.3, 'k_apa': 0.37, 'kapa': 3e-6, 'k_oms': 0.3134, 'k_tsox': 1000, 'k_FeSpre': 0.001, 'accel': 30, 'f_pfe': 1e-6, 'k_pdesorb_c': 1.35, 'Cx1': 112, 'Ny1': 10, 'Pz1': 1, 'Cx2': 200, 'Ny2': 20, 'Pz2': 1}

    sediment.rates['R1a'] = 'accel * k_OM1*OM1 * O2 /  (Km_O2 + O2)'
    sediment.rates['R1b'] = 'accel * k_OM2*OM2 * O2 /  (Km_O2 + O2)'
    sediment.rates['R2a'] = 'k_OM1*OM1 * NO3 /  (Km_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R2b'] = 'k_OM2*OM2 * NO3 /  (Km_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R3a'] = 'k_OM1*OM1 * FeOH3 /  (Km_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R3b'] = 'k_OM2 *OM2 * FeOH3 /  (Km_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R4a'] = 'k_OM1*OM1 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R4b'] = 'k_OM2*OM2 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R5a'] = 'k_OM1*OM1 * SO4 / (Km_SO4 + SO4 ) * Kin_FeOOH / (Kin_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R5b'] = 'k_OM2*OM2 * SO4 / (Km_SO4 + SO4 ) * Kin_FeOOH / (Kin_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
    sediment.rates['R6'] = 'k_tsox * O2 * (HS+H2S)'
    sediment.rates['R7'] = 'k_tS_Fe * FeOH3 *  (HS+H2S)'
    sediment.rates['R8'] = 'k_Feox * Fe2 * O2'
    sediment.rates['R9'] = 'k_amox * O2 / (Km_oxao + O2) * (NH4 / (Km_amao + NH4))'
    sediment.rates['R10'] = 'k_oms * (HS+H2S) * (OM1 + OM2)'
    sediment.rates['R11'] = 'k_FeSpre * FeS * S0'
    sediment.rates['R12'] = 'k_rhom * O2 * FeS'
    sediment.rates['R13'] = 'k_FeS2pre * FeS * (HS+H2S)'
    sediment.rates['R14a'] = 'k_Fe_pre * ( Fe2 * (HS+H2S) / (1e-3**2 * Ks_FeS) - 1)'
    sediment.rates['R14b'] = 'k_Fe_dis * FeS * ( 1 - Fe2 * (HS+H2S) / (1e-3**2 * Ks_FeS))'
    sediment.rates['R15a'] = 'k_Spre * S0'
    sediment.rates['R15b'] = 'k_Sdis * S8'
    sediment.rates['R16a'] = 'k_pdesorb_a * FeOH3 * PO4'
    sediment.rates[
        'R16b'] = 'f_pfe * (4 * (k_OM1*OM1 * FeOH3 /  (Km_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)+k_OM2*OM2 * NO3 /  (Km_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)) + 2 * k_tS_Fe * FeOH3 *  (HS+H2S))'
    sediment.rates['R17a'] = 'k_pdesorb_b * FeOOH * PO4'
    sediment.rates['R17b'] = 'f_pfe * (4 * (k_OM1*OM1 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)+k_OM2*OM2 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)))'
    sediment.rates['R18a'] = 'k_pdesorb_c * PO4 * AlOH3'
    sediment.rates['R19'] = 'k_apa * (PO4 - kapa)'

    sediment.dcdt['O2'] = '-0.25 * R8  - 2 * R9  - (R1a+R1b) - 3 * R12'
    sediment.dcdt['OM1'] = '-1/Cx1*(R1a+R2a+R3a+R4a+R5a) - R10 '
    sediment.dcdt['OM2'] = '-1/Cx2*(R1b+R2b+R3b+R4b+R5b) - R10'
    sediment.dcdt['NO3'] = '- 0.8*(R2a+R2b)+ R9'
    sediment.dcdt['FeOH3'] = '-4 * (R3a+R3b) - R16a - 2*R7 + R8'
    sediment.dcdt['SO4'] = '- 0.5*(R5a+R5b) + R6'
    sediment.dcdt['NH4'] = '(Ny1/Cx1 * (R1a+R2a+R3a+R4a+R5a) + Ny2/Cx2 * (R1b+R2b+R3b+R4b+R5b)) - R9'
    sediment.dcdt['Fe2'] = '4*(R3a+R3b) + 4*(R4a+R4b) + 2*R7 - R8 + R14b - R14a'
    sediment.dcdt['FeOOH'] = '-4*(R4a+R4b) - R17a + R12'
    sediment.dcdt['H2S'] = '0'
    sediment.dcdt['HS'] = '0.5*(R5a+R5b) - R6 - R7 + R14b - R14a - R10 -R13'
    sediment.dcdt['FeS'] = '- R14b - R11 - 4*R12 -R13 + R14a'
    sediment.dcdt['S0'] = '- R11 - R15a + R7 + R15b'
    sediment.dcdt['PO4'] = '(Pz1/Cx1 * (R1a+R2a+R3a+R4a+R5a) + Pz2/Cx2 * (R1b+R2b+R3b+R4b+R5b)) + R16b + R17b - 2 * R19 - R18a - R16a - R17a'
    sediment.dcdt['S8'] = '4*R12 - R15b + R15a'
    sediment.dcdt['FeS2'] = '+ R11 + R13'
    sediment.dcdt['AlOH3'] = '-R18a'
    sediment.dcdt['PO4adsa'] = 'R16a - R16b'
    sediment.dcdt['PO4adsb'] = 'R17a - R17b'
    sediment.dcdt['Ca2'] = '-3*R19'
    sediment.dcdt['Ca3PO42'] = 'R19'
    sediment.dcdt['OMS'] = 'R10'

    for i in np.arange(1, len(time)):
        bc = 0.231 + 0.2 * np.sin(time[i] * 2 * 3.14)
        sediment.new_boundary_condition('O2', bc)
        sediment.integrate_one_timestep(i)

    return sediment


if __name__ == '__main__':
    transport_equation_test()
