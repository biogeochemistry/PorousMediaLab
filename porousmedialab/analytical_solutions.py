from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from porousmedialab.column import Column
import porousmedialab.desolver as desolver

sns.set_style("whitegrid")


def transport_equation_boundary_effect():
    '''Check the transport equation integrator'''

    w = 5
    tend = 5
    dx = 0.1
    length = 30
    phi = 1
    dt = 0.001
    lab = Column(length, dx, tend, dt, w)
    D = 5
    lab.add_species(
        phi,
        'O2',
        D,
        0,
        bc_top_value=1,
        bc_top_type='dirichlet',
        bc_bot_value=0,
        bc_bot_type='flux')
    lab.solve()
    x = np.linspace(0, lab.length, int(lab.length / lab.dx) + 1)
    sol = 1 / 2 * (
        special.erfc((x - lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)) +
        np.exp(lab.w * x / D) * special.erfc(
            (x + lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)))

    plt.figure()
    plt.plot(x, sol, 'k', label='Analytical solution')
    plt.scatter(
        lab.x[::10],
        lab.species['O2'].concentration[:, -1][::10],
        marker='x',
        label='Numerical')
    plt.xlim([x[0], x[-1]])
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    ax.grid(linestyle='-', linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def transport_equation_plot():
    '''Check the transport equation integrator'''

    w = 5
    tend = 5
    dx = 0.1
    length = 100
    phi = 1
    dt = 0.001
    lab = Column(length, dx, tend, dt, w)
    D = 5
    lab.add_species(
        phi,
        'O2',
        D,
        0,
        bc_top_value=1,
        bc_top_type='dirichlet',
        bc_bot_value=0,
        bc_bot_type='dirichlet')
    lab.solve()
    x = np.linspace(0, lab.length, int(lab.length / lab.dx) + 1)
    sol = 1 / 2 * (
        special.erfc((x - lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)) +
        np.exp(lab.w * x / D) * special.erfc(
            (x + lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)))

    plt.figure()
    plt.plot(x, sol, 'k', label='Analytical solution')
    plt.scatter(
        lab.x[::10],
        lab.species['O2'].concentration[:, -1][::10],
        marker='x',
        label='Numerical')
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
    time = np.linspace(0, T, int(T / dt) + 1)
    num_sol = np.array(C0['C'])
    for i in range(1, len(time)):
        C_new, _, _ = desolver.ode_integrate(
            C0, dcdt, rates, coef, dt, solver='rk4')
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
