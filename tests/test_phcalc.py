"""Tests for pH calculation module (Acid, Neutral, System classes)."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from porousmedialab.phcalc import Acid, Neutral, System


class TestNeutral:
    """Tests for the Neutral ion class."""

    def test_neutral_requires_charge(self):
        """Neutral ion must have charge defined."""
        with pytest.raises(ValueError, match="charge.*must be defined"):
            Neutral(conc=0.1)

    def test_neutral_stores_charge(self):
        """Neutral ion should store the charge."""
        ion = Neutral(charge=-1, conc=0.1)
        assert ion.charge == -1
        assert ion.conc == 0.1

    def test_neutral_alpha_always_one(self):
        """Alpha for neutral ion is always 1.0 at any pH."""
        ion = Neutral(charge=1, conc=0.1)
        # Single pH value
        alpha = ion.alpha(7.0)
        assert_allclose(alpha, [[1.0]])

        # Array of pH values
        phs = np.array([1.0, 4.0, 7.0, 10.0, 14.0])
        alphas = ion.alpha(phs)
        assert_allclose(alphas, np.ones((5, 1)))

    def test_neutral_positive_ion(self):
        """Test a positive neutral ion (e.g., K+)."""
        k_plus = Neutral(charge=1, conc=0.05)
        assert k_plus.charge == 1
        assert k_plus.alpha(7.0).shape == (1, 1)

    def test_neutral_negative_ion(self):
        """Test a negative neutral ion (e.g., Cl-)."""
        cl_minus = Neutral(charge=-1, conc=0.1)
        assert cl_minus.charge == -1


class TestAcid:
    """Tests for the Acid class."""

    def test_acid_requires_ka_or_pka(self):
        """Acid must have Ka or pKa defined."""
        with pytest.raises(ValueError, match="Ka or pKa"):
            Acid(charge=0, conc=0.1)

    def test_acid_requires_charge(self):
        """Acid must have charge defined."""
        with pytest.raises(ValueError, match="charge.*must be defined"):
            Acid(pKa=[4.76], conc=0.1)

    def test_acid_pka_to_ka_conversion(self):
        """Verify pKa is correctly converted to Ka."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        expected_ka = 10**(-4.76)
        assert_allclose(acid.Ka, [expected_ka], rtol=1e-10)

    def test_acid_ka_to_pka_conversion(self):
        """Verify Ka is correctly converted to pKa."""
        ka = 1.74e-5  # Acetic acid
        acid = Acid(Ka=[ka], charge=0, conc=0.1)
        expected_pka = -np.log10(ka)
        assert_allclose(acid.pKa, [expected_pka], rtol=1e-10)

    def test_monoprotic_alpha_sum_to_one(self):
        """Alpha fractions must sum to 1.0 for monoprotic acid."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        for ph in [1.0, 4.0, 4.76, 7.0, 10.0, 14.0]:
            alphas = acid.alpha(ph)
            assert_allclose(alphas.sum(), 1.0, rtol=1e-10)

    def test_polyprotic_alpha_sum_to_one(self):
        """Alpha fractions must sum to 1.0 for polyprotic acid."""
        # Phosphoric acid: H3PO4
        acid = Acid(pKa=[2.148, 7.198, 12.375], charge=0, conc=0.1)
        for ph in [1.0, 4.0, 7.0, 10.0, 14.0]:
            alphas = acid.alpha(ph)
            assert_allclose(alphas.sum(), 1.0, rtol=1e-10)

    def test_alpha_at_pka(self):
        """At pH = pKa, adjacent alpha values should be equal."""
        pka = 4.76
        acid = Acid(pKa=[pka], charge=0, conc=0.1)
        alphas = acid.alpha(pka)
        # At pH = pKa, [HA] = [A-], so alpha0 = alpha1 = 0.5
        assert_allclose(alphas[0], 0.5, rtol=0.01)
        assert_allclose(alphas[1], 0.5, rtol=0.01)

    def test_alpha_low_ph_protonated(self):
        """At very low pH, acid should be fully protonated."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        alphas = acid.alpha(0.0)  # Very acidic
        # alpha0 (HA) should be close to 1
        assert alphas[0] > 0.99

    def test_alpha_high_ph_deprotonated(self):
        """At very high pH, acid should be fully deprotonated."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        alphas = acid.alpha(14.0)  # Very basic
        # alpha1 (A-) should be close to 1
        assert alphas[-1] > 0.99

    def test_charge_array_monoprotic(self):
        """Monoprotic acid should have 2 charge states."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        assert len(acid.charge) == 2
        assert_allclose(acid.charge, [0, -1])

    def test_charge_array_polyprotic(self):
        """Polyprotic acid should have n+1 charge states."""
        acid = Acid(pKa=[2.148, 7.198, 12.375], charge=0, conc=0.1)
        assert len(acid.charge) == 4
        assert_allclose(acid.charge, [0, -1, -2, -3])

    def test_pka_sorting(self):
        """pKa values should be sorted in ascending order."""
        # Pass pKa in wrong order
        acid = Acid(pKa=[12.375, 2.148, 7.198], charge=0, conc=0.1)
        assert acid.pKa[0] < acid.pKa[1] < acid.pKa[2]

    def test_alpha_array_input(self):
        """Alpha should work with array of pH values."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        phs = np.array([2.0, 4.76, 7.0, 10.0])
        alphas = acid.alpha(phs)
        assert alphas.shape == (4, 2)
        # Each row should sum to 1
        for i in range(4):
            assert_allclose(alphas[i, :].sum(), 1.0, rtol=1e-10)

    def test_carbonic_acid(self):
        """Test carbonic acid (H2CO3) speciation."""
        # pKa1 = 6.35, pKa2 = 10.33
        acid = Acid(pKa=[6.35, 10.33], charge=0, conc=0.001)
        assert len(acid.charge) == 3  # H2CO3, HCO3-, CO3^2-

        # At pH 7, HCO3- should dominate
        alphas = acid.alpha(8.0)
        assert alphas[1] > alphas[0]  # HCO3- > H2CO3
        assert alphas[1] > alphas[2]  # HCO3- > CO3^2-


class TestSystem:
    """Tests for the acid-base System class."""

    def test_pure_water_ph(self):
        """Pure water should have pH close to 7."""
        system = System()
        system.pHsolve(guess=7.0)
        assert_allclose(system.pH, 7.0, atol=0.1)

    def test_strong_acid_ph(self):
        """Strong acid (high conc, low pKa) should give pH less than 7."""
        # The System class minimizes charge balance difference
        # For acids with charge=1 (like NH4+), it's cationic
        # Use charge=0 for neutral acid (like acetic acid)
        strong_acid = Acid(pKa=[1.0], charge=0, conc=0.1)
        system = System(strong_acid)
        system.pHsolve(guess=1.0, tol=1e-8)
        # pH should be acidic (less than 7)
        assert system.pH < 7.0

    def test_weak_acid_ph(self):
        """Weak acid should give pH between 7 and strong acid pH."""
        # 0.1 M acetic acid
        acetic = Acid(pKa=[4.76], charge=0, conc=0.1)
        system = System(acetic)
        system.pHsolve(guess=3.0)
        # pH should be around 2.87 for 0.1 M acetic acid
        assert 2.5 < system.pH < 3.5

    def test_buffer_solution(self):
        """Buffer solution should have pH close to pKa."""
        # Acetate buffer: acetic acid + sodium acetate
        acetic = Acid(pKa=[4.76], charge=0, conc=0.1)
        sodium = Neutral(charge=1, conc=0.05)  # Na+
        # Adding Na+ shifts equilibrium
        system = System(acetic, sodium)
        system.pHsolve(guess=5.0)
        # pH should be close to pKa for buffer
        assert 4.0 < system.pH < 5.5

    def test_charge_balance(self):
        """System should achieve charge balance at solution pH."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        system = System(acid)
        system.pHsolve()
        # At solution pH, charge balance should be near zero
        diff = system._diff_pos_neg(system.pH)
        assert diff < 1e-4

    def test_guess_estimation(self):
        """Automated guess estimation should work."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        system = System(acid)
        system.pHsolve(guess_est=True, est_num=100)
        assert hasattr(system, 'pH')
        assert 2.0 < system.pH < 5.0

    def test_multiple_acids(self):
        """System with multiple acids should solve correctly."""
        acetic = Acid(pKa=[4.76], charge=0, conc=0.05)
        phosphoric = Acid(pKa=[2.148, 7.198, 12.375], charge=0, conc=0.01)
        system = System(acetic, phosphoric)
        system.pHsolve(guess=4.0)
        # Should converge to a valid pH
        assert 1.0 < system.pH < 7.0

    def test_diff_pos_neg_array(self):
        """_diff_pos_neg should work with array input."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        system = System(acid)
        phs = np.linspace(0, 14, 100)
        diffs = system._diff_pos_neg(phs)
        assert len(diffs) == 100
        # Diffs should have a minimum somewhere
        assert diffs.min() < diffs[0]
        assert diffs.min() < diffs[-1]

    def test_convergence_tolerance(self):
        """Solution should respect tolerance setting."""
        acid = Acid(pKa=[4.76], charge=0, conc=0.1)
        system = System(acid)
        system.pHsolve(tol=1e-8)
        diff = system._diff_pos_neg(system.pH)
        assert diff < 1e-4  # Should be within tolerance


class TestAcidBaseIntegration:
    """Integration tests for acid-base calculations."""

    def test_phosphate_buffer(self):
        """Phosphoric acid system should converge to a solution."""
        # Dihydrogen phosphate / hydrogen phosphate
        # The System finds charge balance
        phosphoric = Acid(pKa=[2.148, 7.198, 12.375], charge=0, conc=0.1)
        system = System(phosphoric)
        system.pHsolve(guess=4.0)
        # Should converge to some pH value (exact value depends on implementation)
        assert hasattr(system, 'pH')
        assert 0.0 < system.pH < 14.0

    def test_henderson_hasselbalch(self):
        """Verify consistency with Henderson-Hasselbalch equation."""
        pKa = 4.76
        acid = Acid(pKa=[pKa], charge=0, conc=0.1)

        # At pH = pKa, ratio [A-]/[HA] = 1
        alphas = acid.alpha(pKa)
        ratio = alphas[1] / alphas[0]
        assert_allclose(ratio, 1.0, rtol=0.01)

        # At pH = pKa + 1, ratio should be 10
        alphas = acid.alpha(pKa + 1)
        ratio = alphas[1] / alphas[0]
        assert_allclose(ratio, 10.0, rtol=0.1)

        # At pH = pKa - 1, ratio should be 0.1
        alphas = acid.alpha(pKa - 1)
        ratio = alphas[1] / alphas[0]
        assert_allclose(ratio, 0.1, rtol=0.1)
