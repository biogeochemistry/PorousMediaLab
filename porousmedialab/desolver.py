import sys
import numexpr as ne
from scipy.sparse import linalg
from scipy.sparse import spdiags
from scipy.integrate import ode


def create_template_AL_AR(phi, diff_coef, adv_coef, bc_top_type, bc_bot_type,
                          dt, dx, N):
    """ creates 2 matrices for transport equation AL and AR

    Args:
        phi (TYPE): vector of porosity(phi) or 1-phi
        diff_coef (float): diffusion coefficient
        adv_coef (float): advection coefficient
        bc_top_type (string): type of boundary condition
        bc_bot_type (string): type of boundary condition
        dt (float): time step
        dx (float): spatial step
        N (int): size of mesh

    Returns:
        array: AL and AR matrices
    """
    # TODO: error source somewhere in non constant
    # porosity profile. Maybe we also need d phi/dx
    s = phi * diff_coef * dt / dx / dx
    q = phi * adv_coef * dt / dx
    AL = spdiags(
        [-s / 2 - q / 4, phi + s, -s / 2 + q / 4], [-1, 0, 1],
        N,
        N,
        format='csr')  # .toarray()
    AR = spdiags(
        [s / 2 + q / 4, phi - s, s / 2 - q / 4], [-1, 0, 1], N, N,
        format='csr')  # .toarray()

    if bc_top_type in ['dirichlet', 'constant']:
        AL[0, 0] = phi[0]
        AL[0, 1] = 0
        AR[0, 0] = phi[0]
        AR[0, 1] = 0
    elif bc_top_type in ['neumann', 'flux']:
        AL[0,0] = phi[0] + s[0]  # + adv_coef * s[0] * dx / diff_coef] - q[0] * adv_coef * dx / diff_coef] / 2
        AL[0, 1] = -s[0]
        AR[0,0] = phi[0] - s[0]  # - adv_coef * s[0] * dx / diff_coef] + q[0] * adv_coef * dx / diff_coef] / 2
        AR[0, 1] = s[0]
    else:
        print('\nABORT!!!: Not correct top boundary condition type...')
        sys.exit()

    if bc_bot_type in ['dirichlet', 'constant']:
        AL[-1, -1] = phi[-1]
        AL[-1, -2] = 0
        AR[-1, -1] = phi[-1]
        AR[-1, -2] = 0
    elif bc_bot_type in ['neumann', 'flux']:
        AL[-1, -1] = phi[-1] + s[-1]
        AL[-1, -2] = -s[-1]  # / 2 - s[-1] / 2
        AR[-1, -1] = phi[-1] - s[-1]
        AR[-1, -2] = s[-1]  # / 2 + s[-1] / 2
    else:
        print('\nABORT!!!: Not correct bottom boundary condition type...')
        sys.exit()
    return AL, AR


def update_matrices_due_to_bc(AR, profile, phi, diff_coef, adv_coef,
                              bc_top_type, bc_top, bc_bot_type, bc_bot, dt, dx,
                              N):
    s = phi * diff_coef * dt / dx / dx
    q = phi * adv_coef * dt / dx

    if (bc_top_type in ['dirichlet', 'constant']
            and bc_bot_type in ['dirichlet', 'constant']):
        profile[0], profile[-1] = bc_top, bc_bot
        B = AR.dot(profile)

    elif (bc_top_type in ['dirichlet', 'constant']
          and bc_bot_type in ['neumann', 'flux']):
        profile[0] = bc_top
        B = AR.dot(profile)
        B[-1] = B[-1] + 2 * 2 * bc_bot * (
            s[-1] / 2 - q[-1] / 4) * dx / phi[-1] / diff_coef

    elif (bc_top_type in ['neumann', 'flux']
          and bc_bot_type in ['dirichlet', 'constant']):
        profile[-1] = bc_bot
        B = AR.dot(profile)
        B[0] = B[0] + 2 * 2 * bc_top * (
            s[0] / 2 - q[0] / 4) * dx / phi[0] / diff_coef

    elif (bc_top_type in ['neumann', 'flux']
          and bc_bot_type in ['neumann', 'flux']):
        B = AR.dot(profile)
        B[0] = B[0] + 2 * 2 * bc_top * (
            s[0] / 2 - q[0] / 4) * dx / phi[0] / diff_coef
        B[-1] = B[-1] + 2 * 2 * bc_bot * (
            s[-1] / 2 - q[-1] / 4) * dx / phi[-1] / diff_coef
    else:
        print('\nABORT!!!: Not correct boundary condition in the species...')
        sys.exit()

    return profile, B


def linear_alg_solver(A, B):
    return linalg.spsolve(A, B, use_umfpack=True)


def ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4'):
    """Integrates the reactions according to 4th Order Runge-Kutta method
    or Butcher 5th where the variables, rates, coef are passed as dictionaries
    """

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

    def k_loop(conc, dt=dt, non_negative_rates=True):
        rates_per_rate = {}
        for element, rate in rates.items():
            rates_per_rate[element] = ne.evaluate(rate, {**coef, **conc})
            if non_negative_rates:
                rates_per_rate[element] = rates_per_rate[element] * (
                    rates_per_rate[element] > 0)

        Kn = {}
        for element in dcdt:
            Kn[element] = dt * ne.evaluate(dcdt[element], {
                **
                coef,
                **
                rates_per_rate
            })

        return Kn, rates_per_rate

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
        k1, rates_per_rate1 = k_loop(C_0)
        k2, rates_per_rate2 = k_loop(sum_k(C_0, k1, 0.5))
        k3, rates_per_rate3 = k_loop(sum_k(C_0, k2, 0.5))
        k4, rates_per_rate4 = k_loop(sum_k(C_0, k3, 1))

        rates_per_rate = {}
        for rate_name, rate in rates_per_rate1.items():
            rates_per_rate[rate_name] = (
                rates_per_rate1[rate_name] + 2 * rates_per_rate2[rate_name] +
                2 * rates_per_rate3[rate_name] + rates_per_rate4[rate_name]) / 6

        C_new = {}
        rates_per_element = {}
        for element in C_0:
            rates_per_element[element] = (
                k1[element] + 2 * k2[element] + 2 * k3[element] + k4[element]
            ) / 6
            C_new[element] = C_0[element] + rates_per_element[element]
        return C_new, rates_per_element, rates_per_rate

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
        k6 = k_loop(
            sum_k(
                sum_k(
                    sum_k(sum_k(sum_k(C_0, k1, -3 / 7), k2, 2 / 7), k3, 12 / 7),
                    k4, -12 / 7), k5, 8 / 7))
        C_new = {}
        rates_per_element = {}
        for element in C_0:
            rates_per_element[element] = (
                7 * k1[element] + 32 * k3[element] + 12 * k4[element] +
                32 * k5[element] + 7 * k6[element]) / 90
            C_new[element] = C_0[element] + rates_per_element[element]
        return C_new, rates_per_element

    if solver == 'butcher5':
        return butcher5(C0)
    if solver == 'implicit':
        return implicit_solver(C0)

    return rk4(C0)


def create_ode_function(species,
                        functions,
                        constants,
                        rates,
                        dcdt,
                        non_negative_rates=True):
    """creates the string of ode function

    Arguments:
        species {dict} -- dict of species provided by user
        constants {dict} -- dict of concstants provided by user
        rates {dict} -- dict of rates provided by user
        dcdt {dict} -- dict of dcdt provided by user

    Keyword Arguments:
        non_negative_rates {bool} -- prevent negative values? (default: {True})

    Returns:
        [str] -- returns string of fun
    """
    body_of_function = "def f(t, y):\n"
    body_of_function += "\t import scipy as sp\n"
    body_of_function += "\t dydt = np.zeros((len(y), 1))"
    for i, s in enumerate(species):
        body_of_function += '\n\t {} = np.clip(y[{:.0f}], 1e-16, 1e+16)'.format(
            s, i)
    for k, v in functions.items():
        body_of_function += '\n\t {} = {}'.format(k, v)
    for k, v in constants.items():
        body_of_function += '\n\t {} = {}'.format(k, v)
    for k, v in rates.items():
        body_of_function += '\n\t {} = {}'.format(k, v, v)
        if non_negative_rates:
            body_of_function += '\n\t {} = {}*({}>0)'.format(k, k, k)
    for i, s in enumerate(dcdt):
        body_of_function += '\n\t dydt[{:.0f}] = {}  # {}'.format(
            i, dcdt[s], s)
    body_of_function += "\n\t return dydt"

    return body_of_function


def create_rate_function(species,
                         functions,
                         constants,
                         rates,
                         dcdt,
                         non_negative_rates=False):
    """creates the string of rates function

    Arguments:
        species {dict} -- dict of species provided by user
        constants {dict} -- dict of concstants provided by user
        rates {dict} -- dict of rates provided by user

    Keyword Arguments:
        non_negative_rates {bool} -- prevent negative values? (default: {True})

    Returns:
        [str] -- returns string of fun
    """
    body_of_function = "def rates(y):\n"
    for i, s in enumerate(species):
        body_of_function += '\n\t {} = np.clip(y[{:.0f}], 1e-16, 1e+16)'.format(
            s, i)
    for k, v in functions.items():
        body_of_function += '\n\t {} = {}'.format(k, v)
    for k, v in constants.items():
        body_of_function += '\n\t {} = {}'.format(k, v)
    for k, v in rates.items():
        body_of_function += '\n\t {} = {}'.format(k, v, v)
        if non_negative_rates:
            body_of_function += '\n\t {} = {}*({}>0)'.format(k, k, k)
    body_of_function += "\n\t return "
    for k, v in rates.items():
        body_of_function += '{}, '.format(k)

    return body_of_function


def create_solver(dydt):
    solver = ode(dydt).set_integrator('lsoda', method='bdf', rtol=1e-2)
    return solver


def ode_integrate_scipy(solver, yinit, timestep):
    t_start = 0.0
    solver.set_initial_value(yinit, t_start)
    while solver.successful() and solver.t < timestep:
        solver.integrate(solver.t + timestep)
    return solver.y
