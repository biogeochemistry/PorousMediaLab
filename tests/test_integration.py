"""Integration tests for multi-species reactive transport simulations.

These tests validate the mathematical correctness of simulations by comparing
against analytical solutions or known conservation laws.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from porousmedialab.batch import Batch
from porousmedialab.column import Column


class TestMassConservation:
    """Tests that verify mass conservation in closed systems."""

    def test_batch_first_order_decay_total_mass(self):
        """In A -> B reaction, total mass (A + B) should be conserved."""
        batch = Batch(tend=10.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        batch.add_species(name='B', init_conc=0.0)
        batch.constants['k'] = 0.5
        batch.rates['decay'] = 'k * A'
        batch.dcdt['A'] = '-decay'
        batch.dcdt['B'] = 'decay'
        batch.solve(verbose=False)

        # Total mass should be conserved
        # Concentration array shape is (N, time_steps) where N=1 for Batch
        total_initial = 1.0  # A=1, B=0
        total_final = batch.species['A']['concentration'][0, -1] + batch.species['B']['concentration'][0, -1]
        assert_allclose(total_final, total_initial, rtol=1e-3)

    def test_batch_reversible_reaction_equilibrium(self):
        """A <-> B should reach equilibrium K = k_f/k_r."""
        batch = Batch(tend=50.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        batch.add_species(name='B', init_conc=0.0)
        batch.constants['kf'] = 0.2  # Forward rate
        batch.constants['kr'] = 0.1  # Reverse rate
        batch.rates['forward'] = 'kf * A'
        batch.rates['reverse'] = 'kr * B'
        batch.dcdt['A'] = '-forward + reverse'
        batch.dcdt['B'] = 'forward - reverse'
        batch.solve(verbose=False)

        # At equilibrium: K = kf/kr = [B]/[A] = 2
        A_final = batch.species['A']['concentration'][0, -1]
        B_final = batch.species['B']['concentration'][0, -1]
        K_observed = B_final / A_final
        K_expected = 0.2 / 0.1  # = 2
        assert_allclose(K_observed, K_expected, rtol=0.05)


class TestAnalyticalSolutions:
    """Tests that compare numerical results to analytical solutions."""

    def test_batch_first_order_decay(self):
        """First-order decay: C(t) = C0 * exp(-k*t)."""
        k = 0.5
        C0 = 1.0
        tend = 5.0

        batch = Batch(tend=tend, dt=0.001)
        batch.add_species(name='A', init_conc=C0)
        batch.constants['k'] = k
        batch.rates['decay'] = 'k * A'
        batch.dcdt['A'] = '-decay'
        batch.solve(verbose=False)

        # Compare with analytical solution
        # Concentration array shape is (N, time_steps) where N=1 for Batch
        # Get all time steps for the single spatial point
        numerical = batch.species['A']['concentration'][0, :]
        times = batch.time
        analytical = C0 * np.exp(-k * times)

        # Should match within 0.1%
        assert_allclose(numerical, analytical, rtol=1e-3)

    def test_batch_second_order_reaction(self):
        """Second-order reaction: dA/dt = -k*A^2, A(t) = A0/(1+k*A0*t)."""
        k = 0.1
        A0 = 2.0
        tend = 10.0
        dt = 0.001

        batch = Batch(tend=tend, dt=dt)
        batch.add_species(name='A', init_conc=A0)
        batch.constants['k'] = k
        batch.rates['r'] = 'k * A * A'
        batch.dcdt['A'] = '-r'
        batch.solve(verbose=False)

        # Compare with analytical solution
        numerical = batch.species['A']['concentration'][0, :]
        times = batch.time
        analytical = A0 / (1 + k * A0 * times)

        assert_allclose(numerical, analytical, rtol=1e-2)


class TestMultiSpeciesSystems:
    """Tests for systems with multiple interacting species."""

    def test_three_species_cascade(self):
        """Test A -> B -> C cascade reaction."""
        batch = Batch(tend=50.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        batch.add_species(name='B', init_conc=0.0)
        batch.add_species(name='C', init_conc=0.0)
        batch.constants['k1'] = 0.3
        batch.constants['k2'] = 0.1
        batch.rates['r1'] = 'k1 * A'
        batch.rates['r2'] = 'k2 * B'
        batch.dcdt['A'] = '-r1'
        batch.dcdt['B'] = 'r1 - r2'
        batch.dcdt['C'] = 'r2'
        batch.solve(verbose=False)

        # Mass conservation: A + B + C = 1
        total = (batch.species['A']['concentration'][0, -1] +
                 batch.species['B']['concentration'][0, -1] +
                 batch.species['C']['concentration'][0, -1])
        assert_allclose(total, 1.0, rtol=1e-3)

        # At long time, almost all should be C
        assert batch.species['C']['concentration'][0, -1] > 0.95

    def test_michaelis_menten_kinetics(self):
        """Test Michaelis-Menten enzyme kinetics."""
        batch = Batch(tend=50.0, dt=0.01)
        batch.add_species(name='S', init_conc=10.0)  # Substrate
        batch.add_species(name='P', init_conc=0.0)   # Product
        batch.constants['Vmax'] = 1.0
        batch.constants['Km'] = 2.0
        batch.rates['r'] = 'Vmax * S / (Km + S)'
        batch.dcdt['S'] = '-r'
        batch.dcdt['P'] = 'r'
        batch.solve(verbose=False)

        # Mass conservation
        total = batch.species['S']['concentration'][0, -1] + batch.species['P']['concentration'][0, -1]
        assert_allclose(total, 10.0, rtol=1e-3)


class TestColumnTransport:
    """Tests for 1D column transport simulations."""

    def test_column_dirichlet_steady_state(self):
        """With constant BCs and no reactions, should reach steady state."""
        col = Column(length=1, dx=0.1, tend=10.0, dt=0.01, w=0)
        col.add_species(
            theta=1.0, name='tracer', D=0.1, init_conc=0,
            bc_top_value=1.0, bc_top_type='dirichlet',
            bc_bot_value=0.0, bc_bot_type='dirichlet'
        )
        col.solve(verbose=False)

        # Should have linear profile at steady state
        # Concentration array shape is (N, time_steps) where N is spatial points
        # Get final time step profile (all spatial points at last time)
        profile = col.species['tracer']['concentration'][:, -1]
        # Linear from 1 at top to 0 at bottom
        expected = 1.0 - col.x / col.length
        assert_allclose(profile, expected, atol=0.01)

    def test_column_advection_pulse(self):
        """Test that advection moves mass downstream."""
        length = 10
        dx = 0.5
        # Number of spatial grid points is int(length/dx) + 1
        n_points = int(length / dx) + 1

        col = Column(length=length, dx=dx, tend=5.0, dt=0.01, w=0.5)
        # Initial pulse in middle
        init_conc = np.zeros(n_points)
        init_conc[5:7] = 1.0
        col.add_species(
            theta=1.0, name='tracer', D=0.01, init_conc=init_conc,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )
        col.solve(verbose=False)

        # Peak should have moved downstream (higher index)
        initial_peak_idx = 6
        final_profile = col.species['tracer']['concentration'][:, -1]
        final_peak_idx = np.argmax(final_profile)
        assert final_peak_idx > initial_peak_idx


class TestNumericalStability:
    """Tests for numerical stability and convergence."""

    def test_timestep_convergence(self):
        """Smaller timesteps should give more accurate results."""
        errors = []
        for dt in [0.1, 0.01, 0.001]:
            batch = Batch(tend=5.0, dt=dt)
            batch.add_species(name='A', init_conc=1.0)
            batch.constants['k'] = 0.5
            batch.rates['r'] = 'k * A'
            batch.dcdt['A'] = '-r'
            batch.solve(verbose=False)

            analytical_final = np.exp(-0.5 * 5.0)
            numerical_final = batch.species['A']['concentration'][0, -1]
            error = abs(numerical_final - analytical_final)
            errors.append(error)

        # Errors should decrease with smaller timestep
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
