import enum
import warnings

import numexpr as ne
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import spdiags
from scipy.integrate import ode, solve_ivp

# Concentration clipping bounds applied inside generated ODE/rate functions to
# keep values within a numerically safe range and avoid overflow/underflow.
CONCENTRATION_CLIP_MIN = 1e-16
CONCENTRATION_CLIP_MAX = 1e+16


def expression_namespace():
    """Namespace available to generated expression functions."""
    return {
        'np': np,
        'ne': ne,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'abs': np.abs,
        'where': np.where,
        '__builtins__': {'len': len, 'range': range},
    }


class InvalidBoundaryConditionError(ValueError):
    """Raised when an invalid boundary condition type is provided."""
    pass


class BoundaryConditionType(enum.Enum):
    """Canonical boundary-condition kinds for the transport solver.

    Each kind has two accepted user-facing aliases (see ``_BC_ALIASES``):
    'constant' is an alias of Dirichlet, 'flux' is an alias of Neumann.
    """
    DIRICHLET = 'dirichlet'
    NEUMANN = 'neumann'


# User-facing boundary-condition type strings (case-insensitive) mapped to the
# canonical kind. Keeps the public string API ('dirichlet'/'constant'/
# 'neumann'/'flux') while letting the solver branch on enum identity.
_BC_ALIASES = {
    'dirichlet': BoundaryConditionType.DIRICHLET,
    'constant': BoundaryConditionType.DIRICHLET,
    'neumann': BoundaryConditionType.NEUMANN,
    'flux': BoundaryConditionType.NEUMANN,
}


def _parse_bc_type(value, descriptor):
    """Resolve a boundary-condition type string to a BoundaryConditionType.

    Arguments:
        value: boundary condition type ('dirichlet', 'constant', 'neumann', or
            'flux'; matched case-insensitively).
        descriptor: boundary name used in the error message, e.g.
            'top boundary' or 'bottom boundary'.

    Returns:
        The matching BoundaryConditionType member.

    Raises:
        InvalidBoundaryConditionError: if value is not a recognized type.
    """
    canonical = _BC_ALIASES.get(str(value).lower())
    if canonical is None:
        raise InvalidBoundaryConditionError(
            f"Invalid {descriptor} condition type: '{value}'. "
            "Valid types are: 'dirichlet', 'constant', 'neumann', 'flux'"
        )
    return canonical


def _validate_bc_combination(bc_top_type, bc_bot_type):
    """Validate and canonicalize a (top, bottom) boundary-condition pair.

    The top boundary is parsed first so an invalid top type is reported before
    an invalid bottom type, preserving the original error precedence.

    Returns:
        Tuple ``(top, bottom)`` of BoundaryConditionType members.

    Raises:
        InvalidBoundaryConditionError: if either type is not recognized.
    """
    top = _parse_bc_type(bc_top_type, "top boundary")
    bot = _parse_bc_type(bc_bot_type, "bottom boundary")
    return top, bot


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

    Raises:
        ValueError: if dx <= 0, phi contains zeros, or diff_coef <= 0
    """
    # Input validation
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if np.any(np.asarray(phi) == 0):
        raise ValueError("Porosity (phi) cannot contain zero values")
    if diff_coef is None or diff_coef <= 0:
        raise ValueError(f"Diffusion coefficient must be positive, got {diff_coef}")

    # TODO: error source somewhere in non constant
    # porosity profile. Maybe we also need d phi/dx
    s = phi * diff_coef * dt / dx / dx

    # Diffusion stability check (Von Neumann criterion)
    cfl = float(np.max(np.abs(np.asarray(s))))
    if cfl > 0.25:
        warnings.warn(
            f"Diffusion stability condition may be violated: max coefficient = {cfl:.4f} > 0.25. "
            "Consider reducing dt or increasing dx.",
            stacklevel=2
        )
    q = phi * adv_coef * dt / dx
    AL = spdiags(
        [-s / 2 - q / 4, phi + s, -s / 2 + q / 4], [-1, 0, 1],
        N,
        N,
        format='csr')  # .toarray()
    AR = spdiags(
        [s / 2 + q / 4, phi - s, s / 2 - q / 4], [-1, 0, 1], N, N,
        format='csr')  # .toarray()

    top_type, bot_type = _validate_bc_combination(bc_top_type, bc_bot_type)

    if top_type is BoundaryConditionType.DIRICHLET:
        AL[0, 0] = phi[0]
        AL[0, 1] = 0
        AR[0, 0] = phi[0]
        AR[0, 1] = 0
    else:  # NEUMANN
        AL[0,0] = phi[0] + s[0]  # + adv_coef * s[0] * dx / diff_coef] - q[0] * adv_coef * dx / diff_coef] / 2
        AL[0, 1] = -s[0]
        AR[0,0] = phi[0] - s[0]  # - adv_coef * s[0] * dx / diff_coef] + q[0] * adv_coef * dx / diff_coef] / 2
        AR[0, 1] = s[0]

    if bot_type is BoundaryConditionType.DIRICHLET:
        AL[-1, -1] = phi[-1]
        AL[-1, -2] = 0
        AR[-1, -1] = phi[-1]
        AR[-1, -2] = 0
    else:  # NEUMANN
        AL[-1, -1] = phi[-1] + s[-1]
        AL[-1, -2] = -s[-1]  # / 2 - s[-1] / 2
        AR[-1, -1] = phi[-1] - s[-1]
        AR[-1, -2] = s[-1]  # / 2 + s[-1] / 2
    return AL, AR


def update_matrices_due_to_bc(AR, profile, phi, diff_coef, adv_coef,
                              bc_top_type, bc_top, bc_bot_type, bc_bot, dt, dx,
                              N):
    s = phi * diff_coef * dt / dx / dx
    q = phi * adv_coef * dt / dx

    top_type, bot_type = _validate_bc_combination(bc_top_type, bc_bot_type)
    top_dirichlet = top_type is BoundaryConditionType.DIRICHLET
    bot_dirichlet = bot_type is BoundaryConditionType.DIRICHLET

    if top_dirichlet and bot_dirichlet:
        profile[0], profile[-1] = bc_top, bc_bot
        B = AR.dot(profile)

    elif top_dirichlet:  # bottom is Neumann
        profile[0] = bc_top
        B = AR.dot(profile)
        B[-1] = B[-1] + 2 * 2 * bc_bot * (
            s[-1] / 2 - q[-1] / 4) * dx / phi[-1] / diff_coef

    elif bot_dirichlet:  # top is Neumann
        profile[-1] = bc_bot
        B = AR.dot(profile)
        B[0] = B[0] + 2 * 2 * bc_top * (
            s[0] / 2 - q[0] / 4) * dx / phi[0] / diff_coef

    else:  # both Neumann
        B = AR.dot(profile)
        B[0] = B[0] + 2 * 2 * bc_top * (
            s[0] / 2 - q[0] / 4) * dx / phi[0] / diff_coef
        B[-1] = B[-1] + 2 * 2 * bc_bot * (
            s[-1] / 2 - q[-1] / 4) * dx / phi[-1] / diff_coef

    return profile, B


def linear_alg_solver(A, B):
    # use_umfpack was dropped: it is a silent no-op unless scikit-umfpack is
    # installed (not a dependency). Cached callers should prefer
    # factorize_transport_matrix(A).solve(B) to avoid re-factorizing per step.
    return linalg.spsolve(A, B)


def factorize_transport_matrix(AL):
    """Pre-factorize the implicit transport matrix ``AL`` once.

    ``AL`` is constant between ``create_template_AL_AR`` rebuilds (only the
    right-hand side ``B`` changes per timestep), so caching its LU factorization
    and reusing ``.solve(B)`` replaces a full ``spsolve`` every step — about a
    10x speedup on the tridiagonal systems used here.

    Returns:
        A SuperLU object whose ``.solve(B)`` applies the cached factorization.
    """
    return linalg.splu(AL.tocsc())


def _k_loop(conc, rates, dcdt, coef, dt, non_negative_rates=True):
    """Evaluate reaction rates and the dt-scaled dcdt increments at ``conc``.

    Returns ``(Kn, rates_per_rate)`` where ``Kn[element]`` is the increment
    ``dt * dcdt`` and ``rates_per_rate`` holds the (optionally clamped
    non-negative) rate values. Shared by the explicit rk4/butcher5 integrators.
    """
    rates_per_rate = {}
    for element, rate in rates.items():
        rates_per_rate[element] = ne.evaluate(rate, {**coef, **conc})
        if non_negative_rates:
            rates_per_rate[element] = rates_per_rate[element] * (
                rates_per_rate[element] > 0)

    Kn = {}
    for element in dcdt:
        Kn[element] = dt * ne.evaluate(dcdt[element], {**coef, **rates_per_rate})

    return Kn, rates_per_rate


def _sum_k(A, B, b):
    """Return ``{k: A[k] + b * B[k]}`` for a Runge-Kutta stage offset.

    ``B`` already carries a factor of ``dt`` from ``_k_loop`` (``Kn = dt *
    ...``), so the stage offset must NOT multiply by ``dt`` again. A previous
    implementation did, which collapsed every RK stage point to a first-order
    perturbation and silently degraded both ``rk4`` and ``butcher5`` to Euler
    accuracy. The ``dt`` parameter was removed so any stray caller fails loudly
    instead of reintroducing the double-``dt`` bug.
    """
    C_new = {}
    for k in A:
        C_new[k] = A[k] + b * B[k]
    return C_new


def _rk4_step(C_0, rates, dcdt, coef, dt):
    """Integrate one timestep with the 4th Order Runge-Kutta method.

        k_1 = dt*dcdt(C0, dt)
        k_2 = dt*dcdt(C0+0.5*k_1, dt)
        k_3 = dt*dcdt(C0+0.5*k_2, dt)
        k_4 = dt*dcdt(C0+k_3, dt)
        C_new = C0 + (k_1+2*k_2+2*k_3+k_4)/6
    """
    k1, rates_per_rate1 = _k_loop(C_0, rates, dcdt, coef, dt)
    k2, rates_per_rate2 = _k_loop(_sum_k(C_0, k1, 0.5), rates, dcdt, coef, dt)
    k3, rates_per_rate3 = _k_loop(_sum_k(C_0, k2, 0.5), rates, dcdt, coef, dt)
    k4, rates_per_rate4 = _k_loop(_sum_k(C_0, k3, 1), rates, dcdt, coef, dt)

    rates_per_rate = {}
    for rate_name in rates_per_rate1:
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


def _butcher5_step(C_0, rates, dcdt, coef, dt):
    """Integrate one timestep with a 5th Order Butcher method.

        k_1 = dt*sediment_rates(C0, dt);
        k_2 = dt*sediment_rates(C0 + 1/4*k_1, dt);
        k_3 = dt*sediment_rates(C0 + 1/8*k_1 + 1/8*k_2, dt);
        k_4 = dt*sediment_rates(C0 - 1/2*k_2 + k_3, dt);
        k_5 = dt*sediment_rates(C0 + 3/16*k_1 + 9/16*k_4, dt);
        k_6 = dt*sediment_rates(C0 - 3/7*k_1 + 2/7*k_2 + 12/7*k_3 - 12/7*k_4 + 8/7*k_5, dt);
        C_new = C0 + (7*k_1 + 32*k_3 + 12*k_4 + 32*k_5 + 7*k_6)/90;
    """
    k1, rpr1 = _k_loop(C_0, rates, dcdt, coef, dt)
    k2, rpr2 = _k_loop(_sum_k(C_0, k1, 1 / 4), rates, dcdt, coef, dt)
    k3, rpr3 = _k_loop(_sum_k(_sum_k(C_0, k1, 1 / 8), k2, 1 / 8), rates, dcdt, coef, dt)
    k4, rpr4 = _k_loop(_sum_k(_sum_k(C_0, k2, -0.5), k3, 1), rates, dcdt, coef, dt)
    k5, rpr5 = _k_loop(_sum_k(_sum_k(C_0, k1, 3 / 16), k4, 9 / 16), rates, dcdt, coef, dt)
    k6, rpr6 = _k_loop(
        _sum_k(
            _sum_k(
                _sum_k(_sum_k(_sum_k(C_0, k1, -3 / 7), k2, 2 / 7), k3, 12 / 7),
                k4, -12 / 7), k5, 8 / 7), rates, dcdt, coef, dt)

    rates_per_rate = {}
    for rate_name in rpr1:
        rates_per_rate[rate_name] = (
            7 * rpr1[rate_name] + 32 * rpr3[rate_name] +
            12 * rpr4[rate_name] + 32 * rpr5[rate_name] +
            7 * rpr6[rate_name]) / 90

    C_new = {}
    rates_per_element = {}
    for element in C_0:
        rates_per_element[element] = (
            7 * k1[element] + 32 * k3[element] + 12 * k4[element] +
            32 * k5[element] + 7 * k6[element]) / 90
        C_new[element] = C_0[element] + rates_per_element[element]
    return C_new, rates_per_element, rates_per_rate


def ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4'):
    """Integrates the reactions according to 4th Order Runge-Kutta method
    or Butcher 5th where the variables, rates, coef are passed as dictionaries
    """
    if solver == 'butcher5':
        return _butcher5_step(C0, rates, dcdt, coef, dt)
    if solver == 'rk4':
        return _rk4_step(C0, rates, dcdt, coef, dt)

    raise ValueError(f"Unknown solver: '{solver}'. Valid options are 'rk4' and 'butcher5'.")


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


def _append_clipped_species(body, species_list, index_expr, allow_negative):
    """Append clipped species-extraction lines to a generated function body.

    ``index_expr(i)`` returns the right-hand-side indexing for species ``i`` as a
    string (e.g. ``'y[0]'`` or ``'y_2d[0, :]'``). Species named in
    ``allow_negative`` keep a symmetric magnitude bound ``[-MAX, MAX]`` so they
    may stay negative (e.g. Temperature); all others keep the non-negative
    concentration floor ``[MIN, MAX]``. Every species still emits an
    ``np.clip(...)`` call for numerical-overflow safety.

    Arguments:
        body: current function body string
        species_list: ordered list of species names
        index_expr: callable mapping species index -> indexing expression string
        allow_negative: set of species names allowed to remain negative

    Returns:
        Updated function body string
    """
    for i, s in enumerate(species_list):
        lower = -CONCENTRATION_CLIP_MAX if s in allow_negative else CONCENTRATION_CLIP_MIN
        body += f'\n\t {s} = np.clip({index_expr(i)}, {lower}, {CONCENTRATION_CLIP_MAX})'
    return body


def _validate_dcdt_species(species, dcdt):
    species_names = set(species)
    dcdt_names = set(dcdt)
    missing = species_names - dcdt_names
    extra = dcdt_names - species_names
    if missing:
        raise ValueError(f"Missing dcdt expressions for species: {sorted(missing)}")
    if extra:
        raise ValueError(f"dcdt contains unknown species: {sorted(extra)}")


def create_ode_function(species,
                        functions,
                        constants,
                        rates,
                        dcdt,
                        non_negative_rates=True,
                        allow_negative=None):
    """Creates the string of ODE function for single-point integration.

    Arguments:
        species: dict of species provided by user
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        dcdt: dict of dcdt provided by user
        non_negative_rates: prevent negative rate values (default True)
        allow_negative: set of species names allowed to remain negative
            (skip the non-negative clip floor); default None means all clipped

    Returns:
        String representation of the ODE function
    """
    _validate_dcdt_species(species, dcdt)
    allow_negative = allow_negative or set()
    species_list = list(species.keys())

    body = "def f(t, y):\n"
    body += "\t dydt = np.zeros(len(y))"

    # Extract species with clipping
    body = _append_clipped_species(
        body, species_list, lambda i: f'y[{i}]', allow_negative)

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Add dcdt assignments
    for i, s in enumerate(species_list):
        body += f'\n\t dydt[{i}] = {dcdt[s]}  # {s}'

    body += "\n\t return dydt"
    return body


def create_vectorized_ode_function(species,
                                   functions,
                                   constants,
                                   rates,
                                   dcdt,
                                   N,
                                   non_negative_rates=True,
                                   allow_negative=None):
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
        allow_negative: set of species names allowed to remain negative
            (skip the non-negative clip floor); default None means all clipped

    Returns:
        String representation of the vectorized ODE function
    """
    num_species = len(species)
    species_list = list(species.keys())
    _validate_dcdt_species(species, dcdt)
    allow_negative = allow_negative or set()

    body = "def f_vectorized(t, y):\n"
    body += f"\t y_2d = y.reshape({num_species}, {N})\n"
    body += f"\t dydt_2d = np.zeros(({num_species}, {N}))\n"

    # Extract species (each becomes array of shape (N,))
    body = _append_clipped_species(
        body, species_list, lambda i: f'y_2d[{i}, :]', allow_negative)

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Add dcdt assignments
    for i, s in enumerate(species_list):
        body += f'\n\t dydt_2d[{i}, :] = {dcdt[s]}  # {s}'

    body += "\n\t return dydt_2d.ravel()"
    return body


def create_rate_function(species,
                         functions,
                         constants,
                         rates,
                         dcdt,
                         non_negative_rates=False,
                         allow_negative=None):
    """Creates the string of rates function for rate reconstruction.

    Arguments:
        species: dict of species provided by user
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        dcdt: dict of dcdt provided by user (unused, kept for API consistency)
        non_negative_rates: prevent negative rate values (default False)
        allow_negative: set of species names allowed to remain negative
            (skip the non-negative clip floor); default None means all clipped

    Returns:
        String representation of the rates function
    """
    allow_negative = allow_negative or set()
    body = "def rates(y):\n"

    # Extract species with clipping
    body = _append_clipped_species(
        body, list(species), lambda i: f'y[{i}]', allow_negative)

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Return all rate values
    body += "\n\t return "
    body += ', '.join(rates.keys())

    return body


def create_vectorized_rate_function(species,
                                    functions,
                                    constants,
                                    rates,
                                    N,
                                    non_negative_rates=False,
                                    allow_negative=None):
    """Creates vectorized rate function for rate reconstruction across N points.

    Processes all N spatial points at once instead of point-by-point.
    Input shape: (num_species, N) - concentrations for all species at all points
    Output shape: (num_rates, N) - rates for all rate expressions at all points

    Arguments:
        species: dict of species provided by user
        functions: dict of functions provided by user
        constants: dict of constants provided by user
        rates: dict of rates provided by user
        N: number of spatial points
        non_negative_rates: prevent negative rate values (default False)
        allow_negative: set of species names allowed to remain negative
            (skip the non-negative clip floor); default None means all clipped

    Returns:
        String representation of the vectorized rates function
    """
    num_species = len(species)
    num_rates = len(rates)
    allow_negative = allow_negative or set()
    species_list = list(species.keys())
    rate_list = list(rates.keys())

    body = "def rates_vectorized(conc_2d):\n"
    body += f"\t # conc_2d shape: ({num_species}, {N})\n"
    body += f"\t # Each species becomes array of shape ({N},)\n"

    # Extract species (each becomes array of shape (N,))
    body = _append_clipped_species(
        body, species_list, lambda i: f'conc_2d[{i}, :]', allow_negative)

    # Add functions, constants, and rates
    body = _append_functions_constants_rates(body, functions, constants, rates, non_negative_rates)

    # Return stacked rate arrays. Each rate is broadcast to shape (N,) first so
    # that a constant-only (scalar) rate can be stacked alongside a
    # species-dependent (vector) rate; without this, np.stack raises on the mixed
    # shapes for a model that mixes zero-order and higher-order rates.
    if num_rates == 1:
        body += f"\n\t return np.broadcast_to({rate_list[0]}, ({N},))"
    else:
        stacked = ', '.join(f'np.broadcast_to({r}, ({N},))' for r in rate_list)
        body += f"\n\t return np.stack([{stacked}], axis=0)"

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
