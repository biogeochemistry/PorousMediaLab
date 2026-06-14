"""Tests for ODE solvers and sparse matrix creation."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import issparse

from porousmedialab.desolver import (
    create_template_AL_AR,
    update_matrices_due_to_bc,
    linear_alg_solver,
    ode_integrate,
    create_ode_function,
    create_rate_function,
    create_solver,
    ode_integrate_scipy,
    ode_integrate_vectorized,
    expression_namespace,
    InvalidBoundaryConditionError,
    ODESolverError
)


class TestCreateTemplateALAR:
    """Tests for the sparse matrix creation function."""

    def test_returns_sparse_matrices(self):
        """AL and AR should be sparse matrices."""
        phi = np.ones(10)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=10
        )
        assert issparse(AL)
        assert issparse(AR)

    def test_matrix_shape(self):
        """AL and AR should be NxN matrices."""
        N = 15
        phi = np.ones(N)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=N
        )
        assert AL.shape == (N, N)
        assert AR.shape == (N, N)

    def test_dirichlet_dirichlet_bc(self):
        """Test Dirichlet BCs at both boundaries."""
        phi = np.ones(5)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=5
        )
        # For Dirichlet, boundary rows should have identity-like structure
        AL_dense = AL.toarray()
        assert AL_dense[0, 1] == 0  # No coupling to neighbor
        assert AL_dense[-1, -2] == 0

    def test_neumann_neumann_bc(self):
        """Test Neumann (flux) BCs at both boundaries."""
        phi = np.ones(5)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='neumann', bc_bot_type='neumann',
            dt=0.01, dx=0.1, N=5
        )
        # For Neumann, boundary rows have different structure
        AL_dense = AL.toarray()
        assert AL_dense[0, 1] != 0  # Coupling exists
        assert AL_dense[-1, -2] != 0

    def test_mixed_bc_dirichlet_neumann(self):
        """Test mixed BCs: Dirichlet top, Neumann bottom."""
        phi = np.ones(5)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='neumann',
            dt=0.01, dx=0.1, N=5
        )
        AL_dense = AL.toarray()
        assert AL_dense[0, 1] == 0  # Dirichlet at top
        assert AL_dense[-1, -2] != 0  # Neumann at bottom

    def test_constant_bc_alias(self):
        """'constant' should be treated same as 'dirichlet'."""
        phi = np.ones(5)
        AL1, AR1 = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=5
        )
        AL2, AR2 = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='constant', bc_bot_type='constant',
            dt=0.01, dx=0.1, N=5
        )
        assert_allclose(AL1.toarray(), AL2.toarray())
        assert_allclose(AR1.toarray(), AR2.toarray())

    def test_flux_bc_alias(self):
        """'flux' should be treated same as 'neumann'."""
        phi = np.ones(5)
        AL1, AR1 = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='neumann', bc_bot_type='neumann',
            dt=0.01, dx=0.1, N=5
        )
        AL2, AR2 = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='flux', bc_bot_type='flux',
            dt=0.01, dx=0.1, N=5
        )
        assert_allclose(AL1.toarray(), AL2.toarray())
        assert_allclose(AR1.toarray(), AR2.toarray())

    def test_invalid_top_bc_raises_error(self):
        """Invalid top BC type should raise InvalidBoundaryConditionError."""
        phi = np.ones(5)
        with pytest.raises(InvalidBoundaryConditionError, match="top boundary"):
            create_template_AL_AR(
                phi, diff_coef=1.0, adv_coef=0.0,
                bc_top_type='invalid', bc_bot_type='dirichlet',
                dt=0.01, dx=0.1, N=5
            )

    def test_invalid_bot_bc_raises_error(self):
        """Invalid bottom BC type should raise InvalidBoundaryConditionError."""
        phi = np.ones(5)
        with pytest.raises(InvalidBoundaryConditionError, match="bottom boundary"):
            create_template_AL_AR(
                phi, diff_coef=1.0, adv_coef=0.0,
                bc_top_type='dirichlet', bc_bot_type='invalid',
                dt=0.01, dx=0.1, N=5
            )

    def test_advection_affects_matrices(self):
        """Non-zero advection should change matrix structure."""
        phi = np.ones(5)
        AL0, AR0 = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=5
        )
        AL1, AR1 = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=1.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=5
        )
        # Matrices should differ when advection is present
        assert not np.allclose(AL0.toarray(), AL1.toarray())

    def test_variable_porosity(self):
        """Variable porosity should be handled correctly."""
        phi = np.linspace(0.5, 1.0, 10)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=10
        )
        # Should not raise an error
        assert AL.shape == (10, 10)


class TestUpdateMatricesDueToBc:
    """Tests for boundary condition application to matrices."""

    def test_dirichlet_dirichlet_updates(self):
        """Test that Dirichlet BC values are applied to profile."""
        phi = np.ones(5)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=5
        )
        profile = np.zeros(5)
        profile, B = update_matrices_due_to_bc(
            AR, profile, phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_top=1.0,
            bc_bot_type='dirichlet', bc_bot=0.5,
            dt=0.01, dx=0.1, N=5
        )
        assert profile[0] == 1.0
        assert profile[-1] == 0.5

    def test_returns_b_vector(self):
        """Update should return a B vector for linear solve."""
        phi = np.ones(5)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=5
        )
        profile = np.ones(5)
        profile, B = update_matrices_due_to_bc(
            AR, profile, phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_top=1.0,
            bc_bot_type='dirichlet', bc_bot=0.0,
            dt=0.01, dx=0.1, N=5
        )
        assert len(B) == 5


class TestLinearAlgSolver:
    """Tests for the sparse linear algebra solver."""

    def test_identity_system(self):
        """Solving Ax=b with A=I should give x=b."""
        from scipy.sparse import eye
        A = eye(5, format='csr')
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = linear_alg_solver(A, b)
        assert_allclose(x, b)

    def test_simple_system(self):
        """Test a simple tridiagonal system."""
        from scipy.sparse import diags
        # Create a simple diagonally dominant matrix
        A = diags([[-1]*4, [4]*5, [-1]*4], [-1, 0, 1], format='csr')
        b = np.ones(5)
        x = linear_alg_solver(A, b)
        # Verify Ax = b
        assert_allclose(A.dot(x), b, rtol=1e-10)


class TestOdeIntegrateRK4:
    """Tests for RK4 ODE integration."""

    def test_exponential_decay(self, simple_ode_system):
        """RK4 should accurately integrate exponential decay dC/dt = -kC."""
        C0 = dict(simple_ode_system['C0'])
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']
        T = simple_ode_system['T']

        time = np.linspace(0, T, int(T / dt) + 1)
        num_sol = np.array([C0['C']])

        for i in range(1, len(time)):
            C_new, _, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')
            C0['C'] = C_new['C']
            num_sol = np.append(num_sol, C_new['C'])

        # Compare to analytical: C(t) = C0 * exp(-k*t)
        analytical = 1.0 * np.exp(-coef['k'] * time)
        assert_allclose(num_sol, analytical, rtol=1e-4)

    def test_rk4_returns_three_values(self, simple_ode_system):
        """RK4 should return C_new, rates_per_element, rates_per_rate."""
        C0 = simple_ode_system['C0']
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']

        result = ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')
        assert len(result) == 3
        C_new, rates_per_elem, rates_per_rate = result
        assert 'C' in C_new
        assert 'C' in rates_per_elem
        assert 'R' in rates_per_rate

    def test_rk4_preserves_keys(self, simple_ode_system):
        """RK4 output should have same keys as input."""
        C0 = simple_ode_system['C0']
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']

        C_new, _, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')
        assert set(C_new.keys()) == set(C0.keys())


class TestOdeIntegrateButcher5:
    """Tests for Butcher 5th order ODE integration."""

    def test_exponential_decay(self, simple_ode_system):
        """Butcher5 should accurately integrate exponential decay."""
        C0 = dict(simple_ode_system['C0'])
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']
        T = simple_ode_system['T']

        time = np.linspace(0, T, int(T / dt) + 1)
        num_sol = np.array([C0['C']])

        for i in range(1, len(time)):
            C_new, _, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver='butcher5')
            C0['C'] = C_new['C']
            num_sol = np.append(num_sol, C_new['C'])

        analytical = 1.0 * np.exp(-coef['k'] * time)
        assert_allclose(num_sol, analytical, rtol=1e-4)

    def test_butcher5_returns_three_values(self, simple_ode_system):
        """Butcher5 should return C_new, rates_per_element, and rates_per_rate."""
        C0 = simple_ode_system['C0']
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']

        result = ode_integrate(C0, dcdt, rates, coef, dt, solver='butcher5')
        assert len(result) == 3
        C_new, rates_per_element, rates_per_rate = result
        assert 'C' in C_new
        assert 'C' in rates_per_element
        assert 'R' in rates_per_rate

    def test_butcher5_rates_per_rate(self, simple_ode_system):
        """Butcher5 rates_per_rate should contain arrays for all rate names."""
        C0 = simple_ode_system['C0']
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']

        _, _, rates_per_rate = ode_integrate(C0, dcdt, rates, coef, dt, solver='butcher5')
        assert set(rates_per_rate.keys()) == set(rates.keys())
        for rate_name in rates_per_rate:
            assert isinstance(rates_per_rate[rate_name], (float, np.floating, np.ndarray))

    def test_butcher5_matches_rk4(self, simple_ode_system):
        """Butcher5 and RK4 should produce similar results for simple decay."""
        C0_b = dict(simple_ode_system['C0'])
        C0_r = dict(simple_ode_system['C0'])
        coef = simple_ode_system['coef']
        rates = simple_ode_system['rates']
        dcdt = simple_ode_system['dcdt']
        dt = simple_ode_system['dt']

        C_b, _, _ = ode_integrate(C0_b, dcdt, rates, coef, dt, solver='butcher5')
        C_r, _, _ = ode_integrate(C0_r, dcdt, rates, coef, dt, solver='rk4')
        assert_allclose(C_b['C'], C_r['C'], rtol=1e-3)


class TestOdeIntegrateMultiSpecies:
    """Tests for ODE integration with multiple species."""

    def test_two_species_decay(self):
        """Test decay of two independent species."""
        C0 = {'A': 1.0, 'B': 2.0}
        coef = {'k1': 1.0, 'k2': 2.0}
        rates = {'R1': 'k1*A', 'R2': 'k2*B'}
        dcdt = {'A': '-R1', 'B': '-R2'}
        dt = 0.0001
        T = 0.01

        time = np.linspace(0, T, int(T / dt) + 1)
        A_sol = [C0['A']]
        B_sol = [C0['B']]

        for i in range(1, len(time)):
            C_new, _, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')
            C0 = C_new
            A_sol.append(C_new['A'])
            B_sol.append(C_new['B'])

        # Compare to analytical
        A_analytical = 1.0 * np.exp(-1.0 * time)
        B_analytical = 2.0 * np.exp(-2.0 * time)
        assert_allclose(A_sol, A_analytical, rtol=1e-4)
        assert_allclose(B_sol, B_analytical, rtol=1e-4)

    def test_coupled_species(self):
        """Test coupled reaction A -> B."""
        C0 = {'A': 1.0, 'B': 0.0}
        coef = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R', 'B': 'R'}
        dt = 0.0001
        T = 0.01

        for i in range(int(T / dt)):
            C_new, _, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')
            C0 = C_new

        # Mass conservation: A + B should stay constant
        assert_allclose(C_new['A'] + C_new['B'], 1.0, rtol=1e-10)


class TestOdeIntegrateKeyError:
    """Tests for error handling in ODE integration."""

    def test_mismatched_keys_raises_error(self):
        """Mismatched keys in dcdt and C0 should raise KeyError."""
        C0 = {'C': 1}
        coef = {'k': 2}
        rates = {'R': 'k*C'}
        dcdt = {'WRONG_KEY': '-R'}  # Wrong key
        dt = 0.0001

        with pytest.raises(KeyError):
            ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4')

    def test_unknown_solver_raises_error(self):
        """Unknown solver name should raise ValueError."""
        C0 = {'C': 1.0}
        coef = {'k': 2}
        rates = {'R': 'k*C'}
        dcdt = {'C': '-R'}
        dt = 0.0001

        with pytest.raises(ValueError, match="Unknown solver"):
            ode_integrate(C0, dcdt, rates, coef, dt, solver='implicit')

    def test_invalid_solver_name_raises_error(self):
        """Completely invalid solver name should raise ValueError."""
        C0 = {'C': 1.0}
        coef = {'k': 2}
        rates = {'R': 'k*C'}
        dcdt = {'C': '-R'}
        dt = 0.0001

        with pytest.raises(ValueError, match="Unknown solver"):
            ode_integrate(C0, dcdt, rates, coef, dt, solver='nonexistent')


class TestCreateOdeFunction:
    """Tests for ODE function string generation."""

    def test_function_string_format(self):
        """Generated function string should be valid Python."""
        species = {'A': None, 'B': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R', 'B': 'R'}

        func_str = create_ode_function(species, functions, constants, rates, dcdt)
        assert 'def f(t, y):' in func_str
        assert 'return dydt' in func_str

    def test_function_includes_species(self):
        """Generated function should include species extraction."""
        species = {'A': None, 'B': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R', 'B': 'R'}

        func_str = create_ode_function(species, functions, constants, rates, dcdt)
        assert 'A = np.clip' in func_str
        assert 'B = np.clip' in func_str

    def test_function_includes_rates(self):
        """Generated function should include rate expressions."""
        species = {'A': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R'}

        func_str = create_ode_function(species, functions, constants, rates, dcdt)
        assert 'R = k*A' in func_str

    def test_non_negative_rates_option(self):
        """non_negative_rates option should add rate clamping."""
        species = {'A': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R'}

        func_str = create_ode_function(
            species, functions, constants, rates, dcdt, non_negative_rates=True)
        assert 'R = R*(R>0)' in func_str

    def test_dcdt_assignment_follows_species_order(self):
        """Generated derivative indexes should not depend on dcdt insertion order."""
        species = {'A': None, 'B': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'B': 'R', 'A': '-R'}

        func_str = create_ode_function(species, functions, constants, rates, dcdt)
        local_vars = {}
        exec(func_str, expression_namespace(), local_vars)
        dydt = local_vars['f'](0, np.array([1.0, 0.0]))

        assert_allclose(dydt, [-1.0, 1.0])

    def test_missing_dcdt_raises_error(self):
        """Every species should have an explicit derivative expression."""
        with pytest.raises(ValueError, match="Missing dcdt"):
            create_ode_function(
                {'A': None, 'B': None},
                {},
                {},
                {},
                {'A': '0'})


class TestCreateRateFunction:
    """Tests for rate function string generation."""

    def test_rate_function_string_format(self):
        """Generated rate function string should be valid Python."""
        species = {'A': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R'}

        func_str = create_rate_function(species, functions, constants, rates, dcdt)
        assert 'def rates(y):' in func_str
        assert 'return' in func_str

    def test_rate_function_returns_rates(self):
        """Generated function should return rate values."""
        species = {'A': None}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R1': 'k*A', 'R2': 'k*A*A'}
        dcdt = {'A': '-R1-R2'}

        func_str = create_rate_function(species, functions, constants, rates, dcdt)
        assert 'return R1,' in func_str or 'return R2,' in func_str


class TestCreateSolver:
    """Tests for scipy ODE solver creation."""

    def test_creates_solver_object(self):
        """create_solver should return a scipy ODE solver."""
        def dydt(t, y):
            return -y
        solver = create_solver(dydt)
        assert solver is not None
        assert hasattr(solver, 'integrate')


class TestOdeIntegrateScipy:
    """Tests for scipy-based ODE integration."""

    def test_exponential_decay(self):
        """Scipy solver should handle exponential decay."""
        def dydt(t, y):
            return -2.0 * y
        solver = create_solver(dydt)
        yinit = np.array([1.0])
        timestep = 0.1

        y_final = ode_integrate_scipy(solver, yinit, timestep)

        # After t=0.1 with dy/dt=-2y, y should be exp(-0.2)
        expected = np.exp(-0.2)
        assert_allclose(y_final[0], expected, rtol=0.01)


class TestVectorizedODE:
    """Tests for vectorized ODE solver."""

    def test_vectorized_ode_function_generation(self):
        """Test that vectorized ODE function is generated correctly."""
        from porousmedialab.desolver import create_vectorized_ode_function

        species = {'A': {}, 'B': {}}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k * A'}
        dcdt = {'A': '-R', 'B': 'R'}
        N = 10

        fun_str = create_vectorized_ode_function(
            species, functions, constants, rates, dcdt, N)

        assert 'f_vectorized' in fun_str
        assert 'y.reshape(2, 10)' in fun_str
        assert 'dydt_2d.ravel()' in fun_str

    def test_vectorized_ode_integration(self):
        """Test that vectorized ODE integration works correctly."""
        from porousmedialab.desolver import (
            create_vectorized_ode_function,
            ode_integrate_vectorized
        )

        species = {'A': {}}
        functions = {}
        constants = {'k': 1.0}
        rates = {'R': 'k * A'}
        dcdt = {'A': '-R'}
        N = 5

        fun_str = create_vectorized_ode_function(
            species, functions, constants, rates, dcdt, N)
        local_vars = {'np': np}
        exec(fun_str, {'np': np}, local_vars)
        f = local_vars['f_vectorized']

        # Initial concentrations (all spatial points start at 1.0)
        yinit = np.ones(N)
        timestep = 0.1

        ynew = ode_integrate_vectorized(f, yinit, timestep)

        # After t=0.1 with dy/dt = -A, A should be approximately exp(-0.1)
        expected = np.exp(-0.1)
        assert_allclose(ynew, expected * np.ones(N), rtol=0.01)

    def test_vectorized_matches_sequential(self):
        """Verify vectorized and sequential produce same results."""
        from porousmedialab.column import Column

        # Run vectorized (default scipy)
        col1 = Column(length=5, dx=0.5, tend=0.1, dt=0.01, ode_method='scipy')
        col1.add_species(theta=1, name='A', D=0.1, init_conc=1.0,
                         bc_top_value=1, bc_top_type='dirichlet',
                         bc_bot_value=0, bc_bot_type='flux')
        col1.constants['k'] = 1.0
        col1.rates['R'] = 'k * A'
        col1.dcdt['A'] = '-R'
        col1.solve(verbose=False)
        result_vec = col1.species['A']['concentration'].copy()

        # Run sequential for comparison
        col2 = Column(length=5, dx=0.5, tend=0.1, dt=0.01, ode_method='scipy_sequential')
        col2.add_species(theta=1, name='A', D=0.1, init_conc=1.0,
                         bc_top_value=1, bc_top_type='dirichlet',
                         bc_bot_value=0, bc_bot_type='flux')
        col2.constants['k'] = 1.0
        col2.rates['R'] = 'k * A'
        col2.dcdt['A'] = '-R'
        col2.solve(verbose=False)
        result_seq = col2.species['A']['concentration'].copy()

        # Different solvers (BDF vs lsoda) have slightly different results
        np.testing.assert_allclose(result_vec, result_seq, rtol=1e-3)

    def test_vectorized_multi_species(self):
        """Test vectorized solver with multiple species."""
        from porousmedialab.column import Column

        col = Column(length=5, dx=0.5, tend=0.1, dt=0.01, ode_method='scipy')
        col.add_species(theta=0.9, name='O2', D=1.0, init_conc=1.0,
                        bc_top_value=1, bc_top_type='dirichlet',
                        bc_bot_value=0, bc_bot_type='flux')
        col.add_species(theta=0.9, name='CH4', D=0.5, init_conc=1.0,
                        bc_top_value=0, bc_top_type='flux',
                        bc_bot_value=1, bc_bot_type='dirichlet')
        col.constants['k'] = 0.5
        col.rates['R'] = 'k * O2 * CH4'
        col.dcdt['O2'] = '-R'
        col.dcdt['CH4'] = '-R'

        # Should not raise any errors
        col.solve(verbose=False)

        # Both concentrations should remain positive
        assert np.all(col.species['O2']['concentration'] >= 0)
        assert np.all(col.species['CH4']['concentration'] >= 0)


class TestODESolverError:
    """Tests for ODESolverError exception."""

    def test_ode_solver_error_is_runtime_error(self):
        """ODESolverError should be a subclass of RuntimeError."""
        assert issubclass(ODESolverError, RuntimeError)

    def test_ode_solver_error_can_be_raised(self):
        """ODESolverError should be raiseable with a message."""
        with pytest.raises(ODESolverError, match="test message"):
            raise ODESolverError("test message")


class TestInputValidation:
    """Tests for input validation in create_template_AL_AR."""

    def test_create_template_raises_on_negative_dx(self):
        """Test that dx < 0 raises ValueError."""
        phi = np.ones(10)
        with pytest.raises(ValueError, match="dx must be positive"):
            create_template_AL_AR(
                phi, diff_coef=1e-5, adv_coef=0.1,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=0.1, dx=-1, N=10
            )

    def test_create_template_raises_on_zero_dx(self):
        """Test that dx = 0 raises ValueError."""
        phi = np.ones(10)
        with pytest.raises(ValueError, match="dx must be positive"):
            create_template_AL_AR(
                phi, diff_coef=1e-5, adv_coef=0.1,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=0.1, dx=0, N=10
            )

    def test_create_template_raises_on_zero_porosity(self):
        """Test that phi containing zero raises ValueError."""
        phi = np.array([0.5, 0, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="Porosity.*cannot contain zero"):
            create_template_AL_AR(
                phi, diff_coef=1e-5, adv_coef=0.1,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=0.1, dx=1, N=5
            )

    def test_create_template_raises_on_negative_diffusion(self):
        """Test that negative diffusion coefficient raises ValueError."""
        phi = np.ones(5)
        with pytest.raises(ValueError, match="Diffusion coefficient must be positive"):
            create_template_AL_AR(
                phi, diff_coef=-1e-5, adv_coef=0.1,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=0.1, dx=1, N=5
            )

    def test_create_template_raises_on_zero_diffusion(self):
        """Test that zero diffusion coefficient raises ValueError."""
        phi = np.ones(5)
        with pytest.raises(ValueError, match="Diffusion coefficient must be positive"):
            create_template_AL_AR(
                phi, diff_coef=0, adv_coef=0.1,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=0.1, dx=1, N=5
            )

    def test_cfl_warning_issued(self):
        """Test that CFL warning is issued when coefficient > 0.25."""
        import warnings
        phi = np.ones(5)
        # Create parameters that will violate CFL: s = phi * diff_coef * dt / dx^2
        # With phi=1, diff_coef=1, dt=1, dx=0.1: s = 1 * 1 * 1 / 0.01 = 100 >> 0.25
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_template_AL_AR(
                phi, diff_coef=1.0, adv_coef=0.0,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=1.0, dx=0.1, N=5
            )
            # Check that a diffusion stability warning was issued
            cfl_warnings = [warning for warning in w if "Diffusion stability" in str(warning.message)]
            assert len(cfl_warnings) > 0, "Expected diffusion stability warning was not issued"

    def test_no_cfl_warning_when_stable(self):
        """Test that no CFL warning is issued when coefficient <= 0.25."""
        import warnings
        phi = np.ones(5)
        # Create stable parameters: s = phi * diff_coef * dt / dx^2
        # With phi=1, diff_coef=0.1, dt=0.01, dx=1: s = 1 * 0.1 * 0.01 / 1 = 0.001 < 0.25
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_template_AL_AR(
                phi, diff_coef=0.1, adv_coef=0.0,
                bc_top_type='dirichlet', bc_bot_type='neumann',
                dt=0.01, dx=1, N=5
            )
            # Check that no CFL warning was issued
            cfl_warnings = [warning for warning in w if "CFL" in str(warning.message)]
            assert len(cfl_warnings) == 0, "Unexpected CFL warning was issued"


class TestBoundaryConditionCombinations:
    """Parametrized coverage of the 4-way (top, bottom) boundary-condition
    combinations exercised through the BoundaryConditionType enum dispatch."""

    BC_COMBINATIONS = [
        ('dirichlet', 'dirichlet'),
        ('dirichlet', 'neumann'),
        ('neumann', 'dirichlet'),
        ('neumann', 'neumann'),
    ]

    @pytest.mark.parametrize("bc_top_type,bc_bot_type", BC_COMBINATIONS)
    def test_create_template_four_way(self, bc_top_type, bc_bot_type):
        """Each combination builds matrices with the correct boundary coupling:
        Dirichlet rows decouple from the neighbour, Neumann rows stay coupled."""
        phi = np.ones(5)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type=bc_top_type, bc_bot_type=bc_bot_type,
            dt=0.01, dx=0.1, N=5
        )
        AL_dense = AL.toarray()
        if bc_top_type == 'dirichlet':
            assert AL_dense[0, 1] == 0
        else:
            assert AL_dense[0, 1] != 0
        if bc_bot_type == 'dirichlet':
            assert AL_dense[-1, -2] == 0
        else:
            assert AL_dense[-1, -2] != 0

    @pytest.mark.parametrize("bc_top_type,bc_bot_type", BC_COMBINATIONS)
    def test_update_matrices_four_way(self, bc_top_type, bc_bot_type):
        """Each combination returns a length-N B vector and writes Dirichlet
        boundary values into the profile endpoints. Explicitly covers the
        neumann/dirichlet and dirichlet/neumann update arms."""
        N = 5
        phi = np.ones(N)
        AL, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type=bc_top_type, bc_bot_type=bc_bot_type,
            dt=0.01, dx=0.1, N=N
        )
        bc_top, bc_bot = 1.0, 0.5
        profile, B = update_matrices_due_to_bc(
            AR, np.zeros(N), phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type=bc_top_type, bc_top=bc_top,
            bc_bot_type=bc_bot_type, bc_bot=bc_bot,
            dt=0.01, dx=0.1, N=N
        )
        assert len(B) == N
        if bc_top_type == 'dirichlet':
            assert profile[0] == bc_top
        if bc_bot_type == 'dirichlet':
            assert profile[-1] == bc_bot

    @pytest.mark.parametrize("alias,canonical", [
        ('constant', 'dirichlet'),
        ('flux', 'neumann'),
    ])
    def test_update_matrices_alias_equivalence(self, alias, canonical):
        """'constant'/'flux' aliases yield identical update output to their
        canonical 'dirichlet'/'neumann' types."""
        N = 5
        phi = np.ones(N)

        def run(bc_type):
            _, AR = create_template_AL_AR(
                phi, diff_coef=1.0, adv_coef=0.0,
                bc_top_type=bc_type, bc_bot_type=bc_type,
                dt=0.01, dx=0.1, N=N
            )
            return update_matrices_due_to_bc(
                AR, np.linspace(1.0, 0.0, N), phi, diff_coef=1.0, adv_coef=0.0,
                bc_top_type=bc_type, bc_top=1.0,
                bc_bot_type=bc_type, bc_bot=0.0,
                dt=0.01, dx=0.1, N=N
            )

        p_alias, B_alias = run(alias)
        p_canon, B_canon = run(canonical)
        assert_allclose(p_alias, p_canon)
        assert_allclose(B_alias, B_canon)

    def test_update_matrices_invalid_type_raises_top_first(self):
        """Invalid BC types passed to update_matrices_due_to_bc raise
        InvalidBoundaryConditionError, reporting the top boundary first."""
        N = 5
        phi = np.ones(N)
        _, AR = create_template_AL_AR(
            phi, diff_coef=1.0, adv_coef=0.0,
            bc_top_type='dirichlet', bc_bot_type='dirichlet',
            dt=0.01, dx=0.1, N=N
        )
        with pytest.raises(InvalidBoundaryConditionError, match="top boundary"):
            update_matrices_due_to_bc(
                AR, np.zeros(N), phi, diff_coef=1.0, adv_coef=0.0,
                bc_top_type='bogus', bc_top=1.0,
                bc_bot_type='also_bogus', bc_bot=0.0,
                dt=0.01, dx=0.1, N=N
            )


class TestOdeIntegrateRefactor:
    """Regression coverage for the ode_integrate module-level helpers
    (_k_loop/_sum_k/_rk4_step/_butcher5_step), using exact / very tight checks.

    The golden values were regenerated after the Tier-4 fix that removed the
    spurious second ``* dt`` in ``_sum_k`` (which had collapsed both solvers to
    first-order Euler accuracy). They are captured exactly from the corrected
    implementation; the companion convergence-order test
    (test_solver_convergence_order) independently validates that rk4/butcher5
    deliver their nominal high order, so a future reintroduction of the bug
    fails even if someone re-captures a golden."""

    # One-step golden outputs captured from the corrected implementation for
    # C0={'C': 1.0}, k=2.0, R='k*C', dcdt={'C': '-R'}, dt=0.01. rk4 matches the
    # exact e^-0.02 = 0.9801986733067553 to its 4th-order truncation residual
    # (~2.7e-11); butcher5 matches to ~1e-14. The two are now distinct (the old
    # shared value 0.9800019998666734 was an artifact of the Euler-collapse bug).
    GOLDEN_C_NEW = {
        'rk4': 0.9801986733333333,
        'butcher5': 0.9801986733067667,
    }

    @pytest.mark.parametrize("solver", ['rk4', 'butcher5'])
    def test_single_step_golden_value(self, solver):
        """A single integration step reproduces the captured golden value."""
        C_new, _, _ = ode_integrate(
            {'C': 1.0}, {'C': '-R'}, {'R': 'k*C'}, {'k': 2.0}, 0.01, solver=solver)
        assert_allclose(float(C_new['C']), self.GOLDEN_C_NEW[solver],
                        rtol=1e-12, atol=0)

    @pytest.mark.parametrize("solver", ['rk4', 'butcher5'])
    def test_deterministic_vector_input(self, solver):
        """Two identical calls on a multi-species vector input return
        bit-identical results (no hidden state or ordering nondeterminism)."""
        def make_inputs():
            C0 = {'A': np.array([1.0, 0.5, 0.2]),
                  'B': np.array([0.0, 0.1, 0.3])}
            return C0, {'A': '-R1', 'B': 'R1-R2'}, \
                {'R1': 'k1*A', 'R2': 'k2*B'}, {'k1': 1.0, 'k2': 0.5}

        C0a, dcdt, rates, coef = make_inputs()
        C0b, _, _, _ = make_inputs()
        res_a = ode_integrate(C0a, dcdt, rates, coef, 0.01, solver=solver)
        res_b = ode_integrate(C0b, dcdt, rates, coef, 0.01, solver=solver)
        for da, db in zip(res_a, res_b):
            assert set(da.keys()) == set(db.keys())
            for key in da:
                np.testing.assert_array_equal(da[key], db[key])

    @pytest.mark.parametrize("solver", ['rk4', 'butcher5'])
    def test_vector_multispecies_matches_analytical_decay(self, solver):
        """Vectorized multi-point integration of independent decay matches the
        analytical exponential solution at each spatial point (both solvers)."""
        C0 = {'A': np.array([1.0, 2.0, 0.5])}
        coef = {'k': 1.0}
        rates = {'R': 'k*A'}
        dcdt = {'A': '-R'}
        dt, T = 0.0001, 0.01
        steps = int(T / dt)
        init = C0['A'].copy()
        for _ in range(steps):
            C0, _, _ = ode_integrate(C0, dcdt, rates, coef, dt, solver=solver)
        analytical = init * np.exp(-coef['k'] * (steps * dt))
        assert_allclose(C0['A'], analytical, rtol=1e-4)

    @pytest.mark.parametrize("solver,min_order",
                             [('rk4', 3.5), ('butcher5', 4.5)])
    def test_solver_convergence_order(self, solver, min_order):
        """Reference-free guard that rk4/butcher5 deliver their nominal high
        order on dC/dt=-kC. This permanently locks out the first-order Euler
        collapse caused by an erroneous extra ``* dt`` in ``_sum_k``: even if a
        golden above were re-captured from buggy code, the measured order would
        drop to ~1 and fail here."""
        k = 2.0
        exact = np.exp(-k)  # integrate from C0=1 over t in [0, 1]

        def final_error(dt):
            C = {'C': np.array([1.0])}
            for _ in range(int(round(1.0 / dt))):
                C, _, _ = ode_integrate(
                    C, {'C': '-R'}, {'R': 'k*C'}, {'k': k}, dt, solver=solver)
            return abs(float(C['C'][0]) - exact)

        measured_order = np.log2(final_error(0.01) / final_error(0.005))
        assert measured_order >= min_order, (
            f"{solver} convergence order {measured_order:.2f} < {min_order} "
            "(possible reintroduction of the _sum_k double-dt Euler bug)")

    # Exact one-step outputs captured from the corrected implementation (Tier-4
    # _sum_k fix) for the nonlinear multi-species vector system below at dt=0.2,
    # where rk4 and butcher5 diverge by ~5e-3 (far above the 1e-12 match
    # tolerance) so the two solvers have DISTINCT goldens. A very tight rtol pins
    # the exact floats while tolerating last-ULP cross-platform noise; a real
    # reorder/drift would be orders of magnitude larger.
    _VEC_GOLDEN = {
        'rk4': {
            'A': [0.7081414362939611, 1.1709248681758635, 0.37450940565395113],
            'B': [0.63698576434565, 0.7141500797468379, 0.9876987081007542],
        },
        'butcher5': {
            'A': [0.7081343485545629, 1.1714497516826388, 0.3745011010849979],
            'B': [0.6368156451852285, 0.7089778108256812, 0.9876764815079293],
        },
    }

    @staticmethod
    def _nonlinear_vector_system():
        """Fresh nonlinear coupled multi-species vector system (3 spatial points)."""
        return (
            {'A': np.array([1.0, 2.0, 0.5]), 'B': np.array([0.5, 0.1, 1.0])},
            {'A': '-R1 - R2', 'B': 'R1 - R2'},
            {'R1': 'k1*A*A', 'R2': 'k2*A*B'},
            {'k1': 1.5, 'k2': 0.8},
        )

    @pytest.mark.parametrize("solver", ['rk4', 'butcher5'])
    def test_vector_golden_matches_reference(self, solver):
        """One step on a nonlinear multi-species vector input reproduces the exact
        per-solver reference output, locking in the post-refactor floats. Because
        the rk4 and butcher5 goldens differ (see next test), this fails if the
        solver dispatch or a stage weight were changed."""
        C0, dcdt, rates, coef = self._nonlinear_vector_system()
        C_new, _, _ = ode_integrate(C0, dcdt, rates, coef, 0.2, solver=solver)
        for species, expected in self._VEC_GOLDEN[solver].items():
            assert_allclose(C_new[species], expected, rtol=1e-12, atol=1e-15)

    def test_rk4_and_butcher5_differ_on_nonlinear_system(self):
        """rk4 and butcher5 produce distinct results on this system, so the golden
        test above is solver-distinguishing (a rk4<->butcher5 dispatch swap fails)."""
        C0r, dcdt, rates, coef = self._nonlinear_vector_system()
        C0b, _, _, _ = self._nonlinear_vector_system()
        rk4_out, _, _ = ode_integrate(C0r, dcdt, rates, coef, 0.2, solver='rk4')
        but_out, _, _ = ode_integrate(C0b, dcdt, rates, coef, 0.2, solver='butcher5')
        max_diff = max(float(np.max(np.abs(rk4_out[k] - but_out[k]))) for k in rk4_out)
        assert max_diff > 1e-9
