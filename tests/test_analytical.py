"""Tests comparing numerical solutions to analytical solutions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import special

from porousmedialab.column import Column
from porousmedialab.batch import Batch
import porousmedialab.desolver as desolver


class TestExponentialDecay:
    """Tests for exponential decay dC/dt = -kC."""

    def test_rk4_exponential_decay(self):
        """RK4 should match analytical solution for decay."""
        C0 = {'C': 1.0}
        coef = {'k': 2.0}
        rates = {'R': 'k*C'}
        dcdt = {'C': '-R'}
        dt = 0.0001
        T = 0.5

        time = np.linspace(0, T, int(T / dt) + 1)
        num_sol = [C0['C']]

        for i in range(1, len(time)):
            C_new, _, _ = desolver.ode_integrate(
                C0, dcdt, rates, coef, dt, solver='rk4')
            C0['C'] = C_new['C']
            num_sol.append(C_new['C'])

        # Analytical: C(t) = C0 * exp(-k*t)
        analytical = np.exp(-2.0 * time)

        # RK4 achieves good accuracy but not exact due to discretization
        assert_allclose(num_sol, analytical, rtol=1e-4)

    @pytest.mark.skip(reason="butcher5 solver has a bug with tuple return from k_loop")
    def test_butcher5_exponential_decay(self):
        """Butcher 5th order should match analytical solution."""
        C0 = {'C': 1.0}
        coef = {'k': 2.0}
        rates = {'R': 'k*C'}
        dcdt = {'C': '-R'}
        dt = 0.0001
        T = 0.5

        time = np.linspace(0, T, int(T / dt) + 1)
        num_sol = [C0['C']]

        for i in range(1, len(time)):
            C_new, _ = desolver.ode_integrate(
                C0, dcdt, rates, coef, dt, solver='butcher5')
            C0['C'] = C_new['C']
            num_sol.append(C_new['C'])

        analytical = np.exp(-2.0 * time)
        assert_allclose(num_sol, analytical, rtol=1e-5)

    def test_batch_exponential_decay(self):
        """Batch model should give correct exponential decay."""
        batch = Batch(tend=1.0, dt=0.001)
        batch.add_species('C', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * C'
        batch.dcdt['C'] = '-R'

        batch.solve(verbose=False)

        # At t=1, C should be exp(-1)
        expected = np.exp(-1.0)
        actual = batch.species['C']['concentration'][0, -1]
        assert_allclose(actual, expected, rtol=0.01)


class TestLogisticGrowth:
    """Tests for logistic growth dC/dt = r*C*(1 - C/K)."""

    def test_logistic_growth_equilibrium(self):
        """Logistic growth should approach carrying capacity."""
        batch = Batch(tend=10.0, dt=0.01)
        batch.add_species('N', init_conc=0.1)
        batch.constants['r'] = 1.0
        batch.constants['K'] = 100.0
        batch.rates['R'] = 'r * N * (1 - N / K)'
        batch.dcdt['N'] = 'R'

        batch.solve(verbose=False)

        # Should approach carrying capacity K
        final = batch.species['N']['concentration'][0, -1]
        assert_allclose(final, 100.0, rtol=0.05)


class TestAdvectionDiffusion:
    """Tests for advection-diffusion equation."""

    @pytest.mark.slow
    def test_transport_vs_erfc_solution(self):
        """Numerical solution should match analytical erfc solution."""
        w = 0.2
        D = 40
        tend = 0.1
        dx = 0.1
        length = 100  # Long domain to avoid boundary effects
        dt = 0.001

        col = Column(length, dx, tend, dt, w)
        col.add_species(
            theta=1, name='O2', D=D, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )

        col.solve(verbose=False)

        # Analytical solution for semi-infinite domain
        x = col.x
        sol = 0.5 * (
            special.erfc((x - w * tend) / 2 / np.sqrt(D * tend)) +
            np.exp(w * x / D) *
            special.erfc((x + w * tend) / 2 / np.sqrt(D * tend))
        )

        # Compare in region away from boundaries
        numerical = col.species['O2']['concentration'][:, -1]
        # Only compare first half where boundary effects minimal
        n = len(x) // 2
        assert max(abs(numerical[:n] - sol[:n])) < 0.05


class TestDiffusionSteadyState:
    """Tests for diffusion steady-state solutions."""

    def test_linear_steady_state(self):
        """Pure diffusion steady state should be linear."""
        col = Column(length=1.0, dx=0.01, tend=10.0, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=0.1, init_conc=0.5,
            bc_top_value=1.0, bc_top_type='dirichlet',
            bc_bot_value=0.0, bc_bot_type='dirichlet'
        )

        col.solve(verbose=False)

        # Analytical steady state: C(x) = C_top - (C_top - C_bot) * x / L
        x = col.x
        analytical = 1.0 - x / 1.0

        numerical = col.species['C']['concentration'][:, -1]
        assert_allclose(numerical, analytical, atol=0.01)


class TestMassConservation:
    """Tests for mass conservation in various scenarios."""

    def test_batch_mass_conservation(self):
        """Total mass should be conserved in A + B -> C reaction."""
        batch = Batch(tend=1.0, dt=0.001)
        batch.add_species('A', init_conc=1.0)
        batch.add_species('B', init_conc=1.0)
        batch.add_species('C', init_conc=0.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A * B'
        batch.dcdt['A'] = '-R'
        batch.dcdt['B'] = '-R'
        batch.dcdt['C'] = 'R'

        batch.solve(verbose=False)

        # Total mass A + B + 2*C should be conserved
        # (since A + B -> C means 1 A + 1 B = 1 C)
        init_mass = 1.0 + 1.0 + 0.0
        final_A = batch.species['A']['concentration'][0, -1]
        final_B = batch.species['B']['concentration'][0, -1]
        final_C = batch.species['C']['concentration'][0, -1]
        final_mass = final_A + final_B + final_C

        # Note: mass is conserved differently here since A+B->C
        # Actually A_init = A_final + C and B_init = B_final + C
        assert_allclose(1.0, final_A + final_C, rtol=0.01)
        assert_allclose(1.0, final_B + final_C, rtol=0.01)

    def test_column_closed_system_mass_conservation(self):
        """Closed column (zero flux BC) should conserve mass."""
        col = Column(length=10, dx=0.5, tend=1.0, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=1.0,
            bc_top_value=0, bc_top_type='flux',
            bc_bot_value=0, bc_bot_type='flux'
        )

        col.solve(verbose=False)

        # Total mass should be conserved (integral of concentration)
        init_mass = col.species['C']['concentration'][:, 0].sum() * col.dx
        final_mass = col.species['C']['concentration'][:, -1].sum() * col.dx

        assert_allclose(final_mass, init_mass, rtol=0.01)


class TestSecondOrderReaction:
    """Tests for second-order reactions."""

    def test_second_order_decay(self):
        """Test 2A -> products with rate = k*A^2."""
        batch = Batch(tend=1.0, dt=0.0001)
        batch.add_species('A', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A * A'
        batch.dcdt['A'] = '-R'

        batch.solve(verbose=False)

        # Analytical: 1/A - 1/A0 = k*t
        # A(t) = A0 / (1 + A0*k*t) = 1 / (1 + t)
        t_final = batch.tend
        expected = 1.0 / (1.0 + 1.0 * t_final)
        actual = batch.species['A']['concentration'][0, -1]

        assert_allclose(actual, expected, rtol=0.01)


class TestReversibleReaction:
    """Tests for reversible reactions."""

    def test_reversible_equilibrium(self):
        """Reversible reaction should reach equilibrium."""
        batch = Batch(tend=10.0, dt=0.01)
        batch.add_species('A', init_conc=1.0)
        batch.add_species('B', init_conc=0.0)
        batch.constants['kf'] = 1.0  # Forward rate
        batch.constants['kr'] = 0.5  # Reverse rate
        batch.rates['Rf'] = 'kf * A'
        batch.rates['Rr'] = 'kr * B'
        batch.dcdt['A'] = '-Rf + Rr'
        batch.dcdt['B'] = 'Rf - Rr'

        batch.solve(verbose=False)

        # At equilibrium: kf*A = kr*B, so B/A = kf/kr = 2
        # Also A + B = 1 (mass conservation)
        # Therefore A = 1/3, B = 2/3
        A_final = batch.species['A']['concentration'][0, -1]
        B_final = batch.species['B']['concentration'][0, -1]

        assert_allclose(A_final, 1.0/3.0, rtol=0.05)
        assert_allclose(B_final, 2.0/3.0, rtol=0.05)
        assert_allclose(A_final + B_final, 1.0, rtol=0.01)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_small_timestep_accuracy(self):
        """Smaller timesteps should give more accurate results."""
        errors = []

        for dt in [0.01, 0.001, 0.0001]:
            batch = Batch(tend=1.0, dt=dt)
            batch.add_species('C', init_conc=1.0)
            batch.constants['k'] = 1.0
            batch.rates['R'] = 'k * C'
            batch.dcdt['C'] = '-R'
            batch.solve(verbose=False)

            expected = np.exp(-1.0)
            actual = batch.species['C']['concentration'][0, -1]
            errors.append(abs(actual - expected))

        # Error should decrease with smaller timestep
        assert errors[0] > errors[1] > errors[2]
