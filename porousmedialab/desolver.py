import numexpr as ne
from scipy.sparse import linalg
from scipy.sparse import spdiags
from scipy.integrate import ode, solve_ivp


class InvalidBoundaryConditionError(ValueError):
    """Raised when an invalid boundary condition type is provided."""
    pass


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
        raise InvalidBoundaryConditionError(
            f"Invalid top boundary condition type: '{bc_top_type}'. "
            "Valid types are: 'dirichlet', 'constant', 'neumann', 'flux'"
        )

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
        raise InvalidBoundaryConditionError(
            f"Invalid bottom boundary condition type: '{bc_bot_type}'. "
            "Valid types are: 'dirichlet', 'constant', 'neumann', 'flux'"
        )
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
        raise InvalidBoundaryConditionError(
            f"Invalid boundary condition combination: top='{bc_top_type}', bot='{bc_bot_type}'. "
            "Valid types are: 'dirichlet', 'constant', 'neumann', 'flux'"
        )

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

        raise NotImplementedError()

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


def _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates):
    """Appends function, constant, and rate definitions to function body string.

    This is a shared helper for ODE function generators.

    Arguments:
        body: current function body string
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        non_negative_rates: if True, clamp negative rates to zero

    Returns:
        Updated function body string
    """
    for k, v in functions.items():
        body += f'\n\t {k} = {v}'
    for k, v in constants.items():
        body += f'\n\t {k} = {v}'
    for k, v in rates.items():
        body += f'\n\t {k} = {v}'
        if non_negative_rates:
            body += f'\n\t {k} = {k}*({k}>0)'
    return body


def create_ode_function(species,
                        functions,
                        constants,
                        rates,
                        dcdt,
                        non_negative_rates=True):
    """Creates the string of ODE function for single-point integration.

    Arguments:
        species: dict of species provided by user
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        dcdt: dict of dcdt provided by user
        non_negative_rates: prevent negative rate values (default True)

    Returns:
        String representation of the ODE function
    """
    body = "def f(t, y):\n"
    body += "\t dydt = np.zeros(len(y))"

    # Extract species with clipping
    for i, s in enumerate(species):
        body += f'\n\t {s} = np.clip(y[{i}], 1e-16, 1e+16)'

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Add dcdt assignments
    for i, s in enumerate(dcdt):
        body += f'\n\t dydt[{i}] = {dcdt[s]}  # {s}'

    body += "\n\t return dydt"
    return body


def create_vectorized_ode_function(species,
                                   functions,
                                   constants,
                                   rates,
                                   dcdt,
                                   N,
                                   non_negative_rates=True):
    """Creates vectorized ODE function handling all N spatial points at once.

    State vector y has shape (N*S,) where S = number of species.
    Internally reshaped to (S, N) for vectorized operations.

    Arguments:
        species: dict of species provided by user
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        dcdt: dict of dcdt provided by user
        N: number of spatial points
        non_negative_rates: prevent negative rate values (default True)

    Returns:
        String representation of the vectorized ODE function
    """
    num_species = len(species)
    species_list = list(species.keys())

    body = "def f_vectorized(t, y):\n"
    body += f"\t y_2d = y.reshape({num_species}, {N})\n"
    body += f"\t dydt_2d = np.zeros(({num_species}, {N}))\n"

    # Extract species (each becomes array of shape (N,))
    for i, s in enumerate(species_list):
        body += f'\n\t {s} = np.clip(y_2d[{i}, :], 1e-16, 1e+16)'

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Add dcdt assignments
    for i, s in enumerate(species_list):
        if s in dcdt:
            body += f'\n\t dydt_2d[{i}, :] = {dcdt[s]}  # {s}'

    body += "\n\t return dydt_2d.ravel()"
    return body


def create_rate_function(species,
                         functions,
                         constants,
                         rates,
                         dcdt,
                         non_negative_rates=False):
    """Creates the string of rates function for rate reconstruction.

    Arguments:
        species: dict of species provided by user
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        dcdt: dict of dcdt provided by user (unused, kept for API consistency)
        non_negative_rates: prevent negative rate values (default False)

    Returns:
        String representation of the rates function
    """
    body = "def rates(y):\n"

    # Extract species with clipping
    for i, s in enumerate(species):
        body += f'\n\t {s} = np.clip(y[{i}], 1e-16, 1e+16)'

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Return all rate values
    body += "\n\t return "
    body += ', '.join(rates.keys())

    return body


def create_solver(dydt):
    solver = ode(dydt).set_integrator('lsoda', method='bdf', rtol=1e-2)
    return solver


def ode_integrate_scipy(solver, yinit, timestep):
    t_start = 0.0
    solver.set_initial_value(yinit, t_start)
    while solver.successful() and solver.t < timestep:
        solver.integrate(solver.t + timestep)
    return solver.y


class ODESolverError(RuntimeError):
    """Raised when the ODE solver fails to converge."""
    pass


def ode_integrate_vectorized(dydt_func, initial_state, timestep, method='LSODA'):
    """Integrates vectorized ODE using scipy.integrate.solve_ivp.

    Arguments:
        dydt_func: vectorized ODE function f(t, y) where y has shape (N*S,)
        initial_state: initial state, shape (N*S,)
        timestep: integration time
        method: solver method (default 'LSODA' which auto-detects stiffness)

    Returns:
        Final state array, shape (N*S,)

    Raises:
        ODESolverError: if the solver fails to converge
    """
    solution = solve_ivp(
        dydt_func,
        t_span=(0.0, timestep),
        y0=initial_state,
        method=method,
        t_eval=[timestep],
        rtol=1e-3,
        atol=1e-6
    )
    if not solution.success:
        raise ODESolverError(
            f"ODE solver failed: {solution.message}. "
            f"Consider adjusting timestep or checking for numerical instabilities."
        )
    return solution.y[:, -1]
