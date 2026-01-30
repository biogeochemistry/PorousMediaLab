"""Tests for equilibrium solver (Henry's Law)."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from porousmedialab.equilibriumsolver import solve_henry_law


class TestHenryLaw:
    """Tests for Henry's Law equilibrium solver."""

    def test_equal_partitioning(self):
        """When Hcc=1, gas and aqueous concentrations should be equal."""
        total_conc = np.array([2.0, 4.0, 6.0])
        Hcc = 1.0
        gas, aq = solve_henry_law(total_conc, Hcc)

        # With Hcc=1: g = totC/(1+1) = totC/2, a = totC - g = totC/2
        assert_allclose(gas, total_conc / 2)
        assert_allclose(aq, total_conc / 2)

    def test_mass_conservation(self):
        """Total concentration should be conserved after partitioning."""
        total_conc = np.array([1.0, 2.5, 5.0, 10.0])
        Hcc = 0.5
        gas, aq = solve_henry_law(total_conc, Hcc)

        assert_allclose(gas + aq, total_conc)

    def test_high_henry_constant(self):
        """High Hcc means most stays in aqueous phase."""
        total_conc = np.array([10.0])
        Hcc = 100.0
        gas, aq = solve_henry_law(total_conc, Hcc)

        # g = totC/(1+100) = totC/101, most remains in aqueous
        assert gas[0] < aq[0]
        assert_allclose(gas, total_conc / 101, rtol=1e-10)

    def test_low_henry_constant(self):
        """Low Hcc means most goes to gas phase."""
        total_conc = np.array([10.0])
        Hcc = 0.01
        gas, aq = solve_henry_law(total_conc, Hcc)

        # g = totC/(1+0.01), most goes to gas
        assert gas[0] > aq[0]
        expected_gas = total_conc / 1.01
        assert_allclose(gas, expected_gas, rtol=1e-10)

    def test_scalar_input(self):
        """Verify solver works with scalar inputs."""
        total_conc = 5.0
        Hcc = 2.0
        gas, aq = solve_henry_law(total_conc, Hcc)

        expected_gas = 5.0 / 3.0  # 5/(1+2)
        expected_aq = 5.0 - expected_gas
        assert_allclose(gas, expected_gas)
        assert_allclose(aq, expected_aq)

    def test_zero_concentration(self):
        """Zero total concentration should give zero for both phases."""
        total_conc = np.array([0.0, 0.0])
        Hcc = 1.0
        gas, aq = solve_henry_law(total_conc, Hcc)

        assert_allclose(gas, [0.0, 0.0])
        assert_allclose(aq, [0.0, 0.0])
