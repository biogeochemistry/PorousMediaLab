import numpy as np
import scipy as sp
from scipy.sparse import spdiags
from scipy import special
import matplotlib.pyplot as plt
# from scikits import umfpack


class Sediment:
    """Sediment class"""

    def __init__(self, L, dx, tend, dt, phi, w):
        super(Sediment, self).__init__()
        self.L = L
        self.dx = dx
        self.tend = tend
        self.dt = dt
        self.phi = phi
        self.w = w
        self.species = {}
        self.dcdt = {}
        self.rates = {}
        self.constants = {}
        self.x = np.linspace(0, L, L / dx + 1)
        self.time = np.linspace(0, tend, tend / dt + 1)
        self.N = self.x.size

    def add_species(self, name, D, init_C, BC_top, is_solute):
        self.species[name] = {}
        self.species[name]['solute?'] = is_solute
        self.species[name]['bc_top'] = BC_top
        self.species[name]['fx_bottom'] = 0
        self.species[name]['D'] = D
        self.species[name]['res'] = np.zeros((self.N, self.time.size))
        self.species[name]['res'][:, 0] = (init_C * np.ones((self.N)))
        self.create_AL_AR(name)
        self.update_bc(name, 0)
        self.dcdt[name] = 0

    def add_solute_species(self, name, D, init_C, BC_top):
        self.add_species(name, D, init_C, BC_top, True)

    def add_solid_species(self, name, D, init_C, BC_top):
        self.add_species(name, D, init_C, BC_top, False)

    def template_AL_AR(self, name, kappa):
        e1 = np.ones((self.N, 1))
        s = kappa * self.species[name]['D'] * self.dt / self.dx / self.dx  #
        q = kappa * self.w * self.dt / self.dx
        self.species[name]['AL'] = spdiags(np.concatenate((e1 * (-s / 2 - q / 4), e1 * (kappa + s), e1 * (-s / 2 + q / 4)), axis=1).T, [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[name]['AR'] = spdiags(np.concatenate((e1 * (s / 2 + q / 4), e1 * (kappa - s), e1 * (s / 2 - q / 4)), axis=1).T, [-1, 0, 1], self.N, self.N, format='csc')  # .toarray()
        self.species[name]['AL'][-1, -1] = kappa + s
        self.species[name]['AL'][-1, -2] = -s
        self.species[name]['AR'][-1, -1] = kappa - s
        self.species[name]['AR'][-1, -2] = s

        if self.species[name]['solute?']:
            self.species[name]['AL'][0, 0] = kappa
            self.species[name]['AL'][0, 1] = 0
            self.species[name]['AR'][0, 0] = kappa
            self.species[name]['AR'][0, 1] = 0
        else:
            self.species[name]['AL'][0, 0] = kappa + s
            self.species[name]['AL'][0, 1] = -s
            self.species[name]['AR'][0, 0] = kappa - s
            self.species[name]['AR'][0, 1] = s

    def create_AL_AR(self, name):
        # create_AL_AR: creates AL and AR matrices
        if self.species[name]['solute?']:
            kappa = self.phi
        else:
            kappa = 1 - self.phi
        self.template_AL_AR(name, kappa)

    def update_bc(self, name, i):
        if self.species[name]['solute?']:
            self.species[name]['res'][0, i] = self.species[name]['bc_top']
            self.species[name]['B'] = self.species[name]['AR'].dot(self.species[name]['res'][:, i])
            # Lower BC is F = w*phi*C
            s = self.phi * self.species[name]['D'] * self.dt / self.dx / self.dx
            self.species[name]['B'][-1] = self.species[name]['B'][-1] - 2 * self.species[name]['res'][-1, i] * self.phi * self.w * s * self.dx / self.phi / self.species[name]['D']
        else:
            self.species[name]['B'] = self.species[name]['AR'].dot(self.species[name]['res'][:, i])
            s = (1 - self.phi) * self.species[name]['D'] * self.dt / self.dx / self.dx
            self.species[name]['B'][0] = self.species[name]['B'][0] + 2 * self.species[name]['bc_top'] * s * self.dx / (1 - self.phi) / self.species[name]['D']
            # Lower BC is F = w*(1-phi)*C
            s = (1 - self.phi) * self.species[name]['D'] * self.dt / self.dx / self.dx
            self.species[name]['B'][-1] = self.species[name]['B'][-1] - 2 * self.species[name]['res'][-1, i] * (1 - self.phi) * self.w * s * self.dx / (1 - self.phi) / self.species[name]['D']

    def solve(self):
        for i in np.arange(1, len(self.time)):
            # self.reactions_integrate(i)
            self.transport_integrate(i)

    def transport_integrate(self, i):
        for name in self.species:
            self.species[name]['res'][:, i] = sp.sparse.linalg.spsolve(self.species[name]['AL'], self.species[name]['B'], use_umfpack=True)
            self.update_bc(name, i)

    # def reactions_integrate(self, name, i):
    #     for name in self.species:
    #         k_1 = dt * sediment.rates(C0, dt)
    #         k_2 = dt * sediment.rates(C0 + 1 / 4 * k_1, dt)
    #         k_3 = dt * sediment.rates(C0 + 1 / 8 * k_1 + 1 / 8 * k_2, dt)
    #         k_4 = dt * sediment.rates(C0 - 1 / 2 * k_2 + k_3, dt)
    #         k_5 = dt * sediment.rates(C0 + 3 / 16 * k_1 + 9 / 16 * k_4, dt)
    #         k_6 = dt * sediment.rates(C0 - 3 / 7 * k_1 + 2 / 7 * k_2 + 12 / 7 * k_3 - 12 / 7 * k_4 + 8 / 7 * k_5, dt)
    #         C0 = C0 + (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) / 90

if __name__ == '__main__':
    D = 40
    w = 1
    t = 1
    dx = 0.1
    L = 10
    phi = 0.9
    dt = 0.001

    # dcdt(:,1)  = -0.25 * R8  - 2 * R9  - R1 * F

    sediment = Sediment(L, dx, t, dt, phi, w)
    sediment.add_solute_species('O2', D, 0.0, 1)
    sediment.add_solid_species('OM', 0.1, 0.0, 1)
    sediment.constants['k'] = 1
    sediment.rates['R1'] = '-k * O2 * OM'
    sediment.dcdt['O2'] = '-4 * R1'
    sediment.dcdt['OM'] = '-R1'
    sediment.solve()

    x = np.linspace(0, L, L / dx + 1)
    sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

    # Bulk concentrations
    plt.figure()
    plt.title('Bulk concentrations')
    plt.plot(sediment.x, phi * sediment.species['O2']['res'][:, -1], 'k')
    plt.plot(sediment.x, (1 - phi) * sediment.species['OM']['res'][:, -1], 'b')

    plt.figure()
    plt.title('Aq and solid concentrations')
    plt.plot(sediment.x, sediment.species['O2']['res'][:, -1], 'k')
    plt.plot(sediment.x, sediment.species['OM']['res'][:, -1], 'b')
    plt.show()
