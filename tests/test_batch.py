"""Tests for Batch reactor (0D) module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from porousmedialab.batch import Batch


class TestBatchInitialization:
    """Tests for Batch class initialization."""

    def test_init_inherits_from_lab(self):
        """Batch should properly inherit from Lab."""
        batch = Batch(tend=1.0, dt=0.01)
        assert batch.tend == 1.0
        assert batch.dt == 0.01
        assert hasattr(batch, 'time')

    def test_init_sets_n_to_one(self):
        """Batch should have N=1 (0-dimensional)."""
        batch = Batch(tend=1.0, dt=0.01)
        assert batch.N == 1

    def test_time_array_correct_length(self):
        """Time array should have correct number of points."""
        batch = Batch(tend=1.0, dt=0.01)
        expected_length = int(1.0 / 0.01) + 1
        assert len(batch.time) == expected_length


class TestBatchAddSpecies:
    """Tests for adding species to Batch."""

    def test_add_species_creates_entry(self):
        """add_species should create species in dict."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        assert 'A' in batch.species

    def test_add_species_sets_initial_concentration(self):
        """add_species should set correct initial concentration."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=5.0)
        assert_allclose(batch.species['A']['concentration'][0, 0], 5.0)

    def test_add_species_creates_concentration_array(self):
        """Species should have concentration array of correct shape."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        # Shape should be (N, time_steps)
        assert batch.species['A']['concentration'].shape == (1, len(batch.time))

    def test_add_species_initializes_profile(self):
        """add_species should initialize profile."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=3.0)
        assert_allclose(batch.profiles['A'], [3.0])

    def test_add_species_sets_default_dcdt(self):
        """add_species should set dcdt to '0' by default."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        assert batch.dcdt['A'] == '0'

    def test_add_species_int_transport_false(self):
        """Batch species should have int_transport=False."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        assert batch.species['A']['int_transport'] is False

    def test_add_multiple_species(self):
        """Should be able to add multiple species."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='A', init_conc=1.0)
        batch.add_species(name='B', init_conc=2.0)
        batch.add_species(name='C', init_conc=0.0)
        assert len(batch.species) == 3


class TestBatchSimpleReactions:
    """Tests for simple reaction simulations."""

    def test_first_order_decay(self):
        """Test first-order decay: dA/dt = -k*A."""
        batch = Batch(tend=1.0, dt=0.001)
        batch.add_species(name='A', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'

        batch.solve(verbose=False)

        # Analytical: A(t) = A0 * exp(-k*t) = exp(-1) at t=1
        expected = np.exp(-1.0)
        actual = batch.species['A']['concentration'][0, -1]
        assert_allclose(actual, expected, rtol=0.01)

    def test_zero_order_production(self):
        """Test zero-order production: dA/dt = k."""
        batch = Batch(tend=1.0, dt=0.001)
        batch.add_species(name='A', init_conc=0.0)
        batch.constants['k'] = 2.0
        batch.rates['R'] = 'k'
        batch.dcdt['A'] = 'R'

        batch.solve(verbose=False)

        # Analytical: A(t) = k*t = 2*1 = 2 at t=1
        expected = 2.0
        actual = batch.species['A']['concentration'][0, -1]
        assert_allclose(actual, expected, rtol=0.01)

    def test_mass_conservation(self):
        """Test mass conservation in A -> B reaction."""
        batch = Batch(tend=1.0, dt=0.001)
        batch.add_species(name='A', init_conc=1.0)
        batch.add_species(name='B', init_conc=0.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'
        batch.dcdt['B'] = 'R'

        batch.solve(verbose=False)

        # A + B should be conserved
        total_init = 1.0 + 0.0
        total_final = (batch.species['A']['concentration'][0, -1] +
                       batch.species['B']['concentration'][0, -1])
        assert_allclose(total_final, total_init, rtol=0.001)


class TestBatchMichaelisMenten:
    """Tests for Michaelis-Menten kinetics."""

    def test_michaelis_menten_saturated(self):
        """At high S, rate should approach Vmax."""
        batch = Batch(tend=0.1, dt=0.0001)
        batch.add_species(name='S', init_conc=100.0)  # High substrate
        batch.add_species(name='P', init_conc=0.0)
        batch.constants['Vmax'] = 10.0
        batch.constants['Km'] = 1.0
        batch.rates['R'] = 'Vmax * S / (Km + S)'
        batch.dcdt['S'] = '-R'
        batch.dcdt['P'] = 'R'

        batch.solve(verbose=False)

        # At S >> Km, dS/dt approx = -Vmax
        # S(t) approx = S0 - Vmax*t = 100 - 10*0.1 = 99
        expected_S = 100.0 - 10.0 * 0.1
        actual_S = batch.species['S']['concentration'][0, -1]
        assert_allclose(actual_S, expected_S, rtol=0.05)


class TestBatchHenryEquilibrium:
    """Tests for Henry's Law equilibrium in batch."""

    def test_henry_equilibrium_partitioning(self):
        """Test that Henry equilibrium partitions correctly."""
        batch = Batch(tend=0.1, dt=0.001)
        batch.add_species(name='O2_aq', init_conc=1.0)
        batch.add_species(name='O2_gas', init_conc=0.0)
        batch.henry_equilibrium('O2_aq', 'O2_gas', Hcc=1.0)

        batch.solve(verbose=False)

        # With Hcc=1, should split equally
        aq_final = batch.species['O2_aq']['concentration'][0, -1]
        gas_final = batch.species['O2_gas']['concentration'][0, -1]
        assert_allclose(aq_final, gas_final, rtol=0.01)
        assert_allclose(aq_final + gas_final, 1.0, rtol=0.01)


class TestBatchAccessSpecies:
    """Tests for accessing species data."""

    def test_dot_notation_access(self):
        """Species should be accessible via batch.species_name."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='O2', init_conc=1.0)
        assert batch.O2 is not None
        assert 'concentration' in batch.O2

    def test_concentration_via_dot_notation(self):
        """Concentration should be accessible via dot notation."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species(name='O2', init_conc=5.0)
        assert_allclose(batch.O2['concentration'][0, 0], 5.0)


class TestBatchPlotMethods:
    """Tests that plot methods exist (smoke tests)."""

    def test_has_plot_method(self):
        """Batch should have plot methods defined."""
        batch = Batch(tend=1.0, dt=0.01)
        assert hasattr(batch, 'plot')
        assert hasattr(batch, 'plot_profiles')
        assert hasattr(batch, 'plot_fractions')
        assert hasattr(batch, 'plot_rates')


class TestBatchAcidBase:
    """Tests for acid-base equilibrium in Batch."""

    def test_create_acid_base_system(self):
        """create_acid_base_system should add pH species."""
        batch = Batch(tend=0.1, dt=0.001)
        batch.add_species(name='HAc', init_conc=0.1)
        batch.add_species(name='Ac', init_conc=0.0)
        batch.add_acid(['HAc', 'Ac'], pKa=[4.76], charge=0)

        batch.create_acid_base_system()

        assert 'pH' in batch.species


class TestBatchReset:
    """Tests for resetting batch simulation."""

    def test_reset_restores_initial(self):
        """Reset should restore initial concentrations for re-run."""
        batch = Batch(tend=0.1, dt=0.001)
        batch.add_species(name='A', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'

        batch.solve(verbose=False)
        final_first_run = batch.species['A']['concentration'][0, -1]

        batch.reset()
        batch.solve(verbose=False)
        final_second_run = batch.species['A']['concentration'][0, -1]

        assert_allclose(final_first_run, final_second_run, rtol=0.001)
