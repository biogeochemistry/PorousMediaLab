"""Tests for Van Genuchten soil hydraulic functions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from porousmedialab.vg import (
    thetaFun, CFun, KFun,
    HygieneSandstone, TouchetSiltLoam, SiltLoamGE3,
    GuelphLoamDrying, GuelphLoamWetting, BeitNetofaClay
)


class TestThetaFun:
    """Tests for the water content function theta(psi)."""

    def test_saturated_at_zero_suction(self, van_genuchten_pars):
        """At psi >= 0, soil should be saturated (theta = thetaS)."""
        pars = van_genuchten_pars
        theta = thetaFun(0.0, pars)
        assert_allclose(theta, pars['thetaS'])

    def test_saturated_at_positive_pressure(self, van_genuchten_pars):
        """At positive psi (ponded), soil should be saturated."""
        pars = van_genuchten_pars
        theta = thetaFun(1.0, pars)
        assert_allclose(theta, pars['thetaS'])

    def test_decreasing_with_suction(self, van_genuchten_pars):
        """Water content should decrease with increasing suction (more negative psi)."""
        pars = van_genuchten_pars
        theta_0 = thetaFun(0.0, pars)
        theta_1 = thetaFun(-1.0, pars)
        theta_5 = thetaFun(-5.0, pars)
        assert theta_0 >= theta_1 >= theta_5

    def test_approaches_residual(self, van_genuchten_pars):
        """At very dry conditions, theta should approach thetaR."""
        pars = van_genuchten_pars
        theta = thetaFun(-1000.0, pars)
        # Should be close to thetaR but not less
        assert theta >= pars['thetaR']
        assert theta < pars['thetaS']

    def test_between_residual_and_saturated(self, van_genuchten_pars):
        """Theta should always be between thetaR and thetaS."""
        pars = van_genuchten_pars
        psi_values = np.linspace(-100, 10, 50)
        for psi in psi_values:
            theta = thetaFun(psi, pars)
            assert pars['thetaR'] <= theta <= pars['thetaS']

    def test_vectorized_input(self, van_genuchten_pars):
        """Function should work with array input."""
        pars = van_genuchten_pars
        psi = np.array([-10, -5, -1, 0, 1])
        theta = thetaFun(psi, pars)
        assert len(theta) == 5


class TestCFun:
    """Tests for the specific moisture capacity C(psi)."""

    def test_capacity_at_saturation(self, van_genuchten_pars):
        """Capacity at saturation should equal Ss (specific storage)."""
        pars = van_genuchten_pars
        # At saturation (Se=1), C should be dominated by Ss term
        C = CFun(0.0, pars)
        # For saturated conditions, Se=1 so first term is 1*Ss
        # Second term involves dSe/dh which is 0 at saturation
        assert_allclose(C, pars['Ss'], rtol=0.01)

    def test_capacity_positive(self, van_genuchten_pars):
        """Specific capacity should always be positive."""
        pars = van_genuchten_pars
        psi_values = np.linspace(-100, 0, 50)
        for psi in psi_values:
            C = CFun(psi, pars)
            assert C >= 0

    def test_capacity_vectorized(self, van_genuchten_pars):
        """Function should work with array input."""
        pars = van_genuchten_pars
        psi = np.array([-10, -5, -1, 0])
        C = CFun(psi, pars)
        assert len(C) == 4


class TestKFun:
    """Tests for the hydraulic conductivity K(psi)."""

    def test_saturated_conductivity(self, van_genuchten_pars):
        """At saturation, K should equal Ks."""
        pars = van_genuchten_pars
        K = KFun(0.0, pars)
        assert_allclose(K, pars['Ks'])

    def test_saturated_conductivity_positive_psi(self, van_genuchten_pars):
        """At positive psi, K should still equal Ks."""
        pars = van_genuchten_pars
        K = KFun(1.0, pars)
        assert_allclose(K, pars['Ks'])

    def test_decreasing_with_suction(self, van_genuchten_pars):
        """K should decrease with increasing suction."""
        pars = van_genuchten_pars
        K_0 = KFun(0.0, pars)
        K_1 = KFun(-1.0, pars)
        K_5 = KFun(-5.0, pars)
        assert K_0 >= K_1 >= K_5

    def test_positive_conductivity(self, van_genuchten_pars):
        """K should always be positive."""
        pars = van_genuchten_pars
        psi_values = np.linspace(-100, 10, 50)
        for psi in psi_values:
            K = KFun(psi, pars)
            assert K >= 0

    def test_k_less_than_ks(self, van_genuchten_pars):
        """Unsaturated K should be less than or equal to Ks."""
        pars = van_genuchten_pars
        psi_values = np.linspace(-100, 0, 50)
        for psi in psi_values:
            K = KFun(psi, pars)
            assert K <= pars['Ks'] * 1.001  # Small tolerance

    def test_vectorized_input(self, van_genuchten_pars):
        """Function should work with array input."""
        pars = van_genuchten_pars
        psi = np.array([-10, -5, -1, 0, 1])
        K = KFun(psi, pars)
        assert len(K) == 5


class TestSoilPresets:
    """Tests for predefined soil parameter sets."""

    def test_hygiene_sandstone_parameters(self):
        """HygieneSandstone should return valid parameters."""
        pars = HygieneSandstone()
        assert pars['thetaR'] < pars['thetaS']
        assert pars['alpha'] > 0
        assert pars['n'] > 1
        assert pars['Ks'] > 0
        assert 'm' in pars
        assert_allclose(pars['m'], 1 - 1 / pars['n'])

    def test_touchet_silt_loam_parameters(self):
        """TouchetSiltLoam should return valid parameters."""
        pars = TouchetSiltLoam()
        assert pars['thetaR'] < pars['thetaS']
        assert pars['n'] > 1
        assert_allclose(pars['m'], 1 - 1 / pars['n'])

    def test_silt_loam_ge3_parameters(self):
        """SiltLoamGE3 should return valid parameters."""
        pars = SiltLoamGE3()
        assert pars['thetaR'] < pars['thetaS']
        assert pars['n'] > 1
        assert_allclose(pars['m'], 1 - 1 / pars['n'])

    def test_guelph_loam_drying_parameters(self):
        """GuelphLoamDrying should return valid parameters."""
        pars = GuelphLoamDrying()
        assert pars['thetaR'] < pars['thetaS']
        assert pars['n'] > 1
        assert_allclose(pars['m'], 1 - 1 / pars['n'])

    def test_guelph_loam_wetting_parameters(self):
        """GuelphLoamWetting should return valid parameters."""
        pars = GuelphLoamWetting()
        assert pars['thetaR'] < pars['thetaS']
        assert pars['n'] > 1
        assert_allclose(pars['m'], 1 - 1 / pars['n'])

    def test_beit_netofa_clay_parameters(self):
        """BeitNetofaClay should return valid parameters."""
        pars = BeitNetofaClay()
        assert pars['thetaR'] <= pars['thetaS']
        assert pars['n'] > 1
        assert_allclose(pars['m'], 1 - 1 / pars['n'])

    def test_hysteresis(self):
        """Drying and wetting curves should differ (hysteresis)."""
        pars_dry = GuelphLoamDrying()
        pars_wet = GuelphLoamWetting()
        # Alpha typically differs between drying and wetting
        assert pars_dry['alpha'] != pars_wet['alpha']
        # thetaS may also differ
        assert pars_dry['thetaS'] != pars_wet['thetaS']


class TestVanGenuchtenPhysics:
    """Physical consistency tests for Van Genuchten model."""

    def test_m_n_relationship(self, van_genuchten_pars):
        """Verify m = 1 - 1/n relationship."""
        pars = van_genuchten_pars
        expected_m = 1 - 1 / pars['n']
        assert_allclose(pars['m'], expected_m)

    def test_effective_saturation_range(self, van_genuchten_pars):
        """Effective saturation should be between 0 and 1."""
        pars = van_genuchten_pars
        psi_values = np.linspace(-100, 10, 100)
        for psi in psi_values:
            theta = thetaFun(psi, pars)
            Se = (theta - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'])
            assert 0 <= Se <= 1.001  # Small tolerance

    def test_water_retention_monotonic(self, van_genuchten_pars):
        """Water retention curve should be monotonically increasing with psi."""
        pars = van_genuchten_pars
        psi = np.linspace(-50, 0, 100)
        theta = thetaFun(psi, pars)
        # Check monotonicity
        differences = np.diff(theta)
        assert np.all(differences >= -1e-10)  # Allow small numerical errors

    def test_conductivity_decreases_orders_of_magnitude(self, van_genuchten_pars):
        """K can decrease by several orders of magnitude."""
        pars = van_genuchten_pars
        K_sat = KFun(0.0, pars)
        K_dry = KFun(-50.0, pars)
        # K should decrease significantly
        assert K_dry < K_sat / 10  # At least one order of magnitude
