"""Tests for the Lab base class."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from porousmedialab.lab import Lab
from porousmedialab.dotdict import DotDict


class TestLabInitialization:
    """Tests for Lab class initialization."""

    def test_init_sets_time_parameters(self):
        """Lab should store tend and dt correctly."""
        lab = Lab(tend=10.0, dt=0.1)
        assert lab.tend == 10.0
        assert lab.dt == 0.1

    def test_init_creates_time_array(self):
        """Lab should create correct time array."""
        lab = Lab(tend=1.0, dt=0.1)
        assert len(lab.time) == 11  # 0.0, 0.1, ..., 1.0
        assert_allclose(lab.time[0], 0.0)
        assert_allclose(lab.time[-1], 1.0)

    def test_init_with_tstart(self):
        """Lab should handle non-zero start time."""
        lab = Lab(tend=2.0, dt=0.1, tstart=1.0)
        assert_allclose(lab.time[0], 1.0)
        assert_allclose(lab.time[-1], 2.0)

    def test_init_creates_empty_containers(self):
        """Lab should initialize empty DotDict containers."""
        lab = Lab(tend=1.0, dt=0.1)
        assert isinstance(lab.species, DotDict)
        assert isinstance(lab.profiles, DotDict)
        assert isinstance(lab.dcdt, DotDict)
        assert isinstance(lab.rates, DotDict)
        assert isinstance(lab.constants, DotDict)
        assert isinstance(lab.functions, DotDict)
        assert len(lab.species) == 0

    def test_init_empty_reactions(self):
        """Lab should start with empty reaction lists."""
        lab = Lab(tend=1.0, dt=0.1)
        assert lab.henry_law_equations == []
        assert lab.acid_base_components == []


class TestLabGetattr:
    """Tests for dot notation access to species."""

    def test_getattr_returns_species(self):
        """lab.species_name should return species dict."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.species['O2'] = DotDict({'concentration': np.zeros(10)})
        o2 = lab.O2
        assert 'concentration' in o2

    def test_getattr_missing_species_raises_keyerror(self):
        """Accessing non-existent species should raise KeyError."""
        lab = Lab(tend=1.0, dt=0.1)
        with pytest.raises(KeyError):
            _ = lab.nonexistent_species


class TestLabHenryEquilibrium:
    """Tests for Henry's Law equilibrium methods."""

    def test_add_partition_equilibrium(self):
        """add_partition_equilibrium should store equation."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.add_partition_equilibrium('O2_aq', 'O2_gas', Hcc=0.03)
        assert len(lab.henry_law_equations) == 1
        eq = lab.henry_law_equations[0]
        assert eq['aq'] == 'O2_aq'
        assert eq['gas'] == 'O2_gas'
        assert eq['Hcc'] == 0.03

    def test_henry_equilibrium_alias(self):
        """henry_equilibrium should be alias for add_partition_equilibrium."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.henry_equilibrium('CO2_aq', 'CO2_gas', Hcc=0.83)
        assert len(lab.henry_law_equations) == 1


class TestLabAcidBase:
    """Tests for acid-base system methods."""

    def test_add_ion(self):
        """add_ion should create a Neutral component."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.add_ion('Cl', charge=-1)
        assert len(lab.acid_base_components) == 1
        component = lab.acid_base_components[0]
        assert component['species'] == ['Cl']
        assert component['pH_object'].charge == -1

    def test_add_acid(self):
        """add_acid should create an Acid component."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.add_acid(['HAc', 'Ac'], pKa=[4.76], charge=0)
        assert len(lab.acid_base_components) == 1
        component = lab.acid_base_components[0]
        assert component['species'] == ['HAc', 'Ac']

    def test_add_polyprotic_acid(self):
        """add_acid should handle polyprotic acids."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.add_acid(
            ['H3PO4', 'H2PO4', 'HPO4', 'PO4'],
            pKa=[2.148, 7.198, 12.375],
            charge=0
        )
        assert len(lab.acid_base_components) == 1


class TestLabReset:
    """Tests for the reset method."""

    def test_reset_restores_initial_profiles(self):
        """reset should restore profiles to initial concentrations."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.species['A'] = DotDict({
            'concentration': np.array([[1.0, 2.0, 3.0]])
        })
        lab.profiles['A'] = np.array([3.0])  # Modified

        lab.reset()

        assert_allclose(lab.profiles['A'], [1.0])


class TestLabConstants:
    """Tests for constants and functions management."""

    def test_add_constants(self):
        """Constants should be accessible via dot notation."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.constants['k'] = 2.0
        lab.constants['Km'] = 0.5
        assert lab.constants.k == 2.0
        assert lab.constants['Km'] == 0.5

    def test_add_functions(self):
        """Functions dict should store string expressions."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.functions['f1'] = 'k * C'
        assert lab.functions.f1 == 'k * C'


class TestLabRates:
    """Tests for rate definitions."""

    def test_add_rates(self):
        """Rates should be stored as string expressions."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.rates['R1'] = 'k * A * B'
        lab.rates['R2'] = 'Vmax * S / (Km + S)'
        assert 'R1' in lab.rates
        assert 'R2' in lab.rates

    def test_add_dcdt(self):
        """dcdt should store concentration change expressions."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.dcdt['A'] = '-R1'
        lab.dcdt['B'] = 'R1 - R2'
        assert lab.dcdt.A == '-R1'
        assert lab.dcdt['B'] == 'R1 - R2'


class TestLabInitRatesArrays:
    """Tests for rate array initialization."""

    def test_init_rates_arrays_creates_zeros(self):
        """init_rates_arrays should create zero arrays for all rates."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.N = 5  # Set spatial dimension
        lab.rates['R1'] = 'k*A'
        lab.rates['R2'] = 'k*B'

        lab.init_rates_arrays()

        assert 'R1' in lab.estimated_rates
        assert 'R2' in lab.estimated_rates
        assert lab.estimated_rates['R1'].shape == (5, len(lab.time))
        assert np.all(lab.estimated_rates['R1'] == 0)


class TestLabOdeMethod:
    """Tests for ODE method selection."""

    def test_default_ode_method(self):
        """Default ODE method should be 'scipy'."""
        lab = Lab(tend=1.0, dt=0.1)
        assert lab.ode_method == 'scipy'

    def test_ode_method_can_be_changed(self):
        """ODE method should be changeable."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.ode_method = 'rk4'
        assert lab.ode_method == 'rk4'


class TestLabEstimateTime:
    """Tests for computation time estimation."""

    def test_estimate_time_at_step_1(self):
        """estimate_time_of_computation should set start time at step 1."""
        lab = Lab(tend=1.0, dt=0.1)
        lab.estimate_time_of_computation(1)
        assert hasattr(lab, 'start_computation_time')

    def test_estimate_time_no_error(self):
        """estimate_time_of_computation should not raise errors."""
        lab = Lab(tend=1.0, dt=0.1)
        # Should not raise at any step
        lab.estimate_time_of_computation(1)
        lab.estimate_time_of_computation(50)
        lab.estimate_time_of_computation(100)
