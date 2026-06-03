"""Tests for Column model (1D advection-diffusion-reaction)."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import special

from porousmedialab.column import Column


class TestColumnInitialization:
    """Tests for Column class initialization."""

    def test_init_sets_domain_parameters(self):
        """Column should store length, dx, tend, dt correctly."""
        col = Column(length=10, dx=0.1, tend=1.0, dt=0.001, w=0.5)
        assert col.length == 10
        assert col.dx == 0.1
        assert col.tend == 1.0
        assert col.dt == 0.001
        assert col.w == 0.5

    def test_init_creates_spatial_grid(self):
        """Column should create correct spatial grid."""
        col = Column(length=10, dx=0.1, tend=1.0, dt=0.001)
        assert len(col.x) == 101  # 0, 0.1, 0.2, ..., 10.0
        assert_allclose(col.x[0], 0.0)
        assert_allclose(col.x[-1], 10.0)

    def test_init_sets_n_from_grid(self):
        """N should be set from spatial grid size."""
        col = Column(length=10, dx=0.5, tend=1.0, dt=0.001)
        assert col.N == 21  # 10/0.5 + 1

    def test_init_rejects_non_divisible_spatial_grid(self):
        """Spatial grid should not silently change the requested dx."""
        with pytest.raises(ValueError, match="space range must be divisible"):
            Column(length=10, dx=3, tend=1.0, dt=0.001)

    def test_init_default_advection_zero(self):
        """Default advection should be 0."""
        col = Column(length=10, dx=0.1, tend=1.0, dt=0.001)
        assert col.w == 0

    def test_init_default_ode_method(self):
        """Default ODE method should be 'scipy'."""
        col = Column(length=10, dx=0.1, tend=1.0, dt=0.001)
        assert col.ode_method == 'scipy'


class TestColumnAddSpecies:
    """Tests for adding species to Column."""

    def test_add_species_creates_entry(self):
        """add_species should create species in dict."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )
        assert 'O2' in col.species

    def test_add_species_sets_diffusion(self):
        """add_species should store diffusion coefficient."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )
        assert col.species['O2']['D'] == 1e-5

    def test_add_species_sets_boundary_conditions(self):
        """add_species should store boundary conditions."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1.5, bc_top_type='dirichlet',
            bc_bot_value=0.5, bc_bot_type='flux'
        )
        assert col.species['O2']['bc_top_value'] == 1.5
        assert col.species['O2']['bc_top_type'] == 'dirichlet'
        assert col.species['O2']['bc_bot_value'] == 0.5
        assert col.species['O2']['bc_bot_type'] == 'flux'

    def test_add_species_bc_type_lowercase(self):
        """Boundary condition types should be converted to lowercase."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='DIRICHLET',
            bc_bot_value=0, bc_bot_type='Flux'
        )
        assert col.species['O2']['bc_top_type'] == 'dirichlet'
        assert col.species['O2']['bc_bot_type'] == 'flux'

    def test_add_species_creates_matrices(self):
        """add_species with int_transport=True should create AL, AR matrices."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux',
            int_transport=True
        )
        assert 'AL' in col.species['O2']
        assert 'AR' in col.species['O2']

    def test_add_species_uniform_porosity(self):
        """Scalar theta should be expanded to array."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001)
        col.add_species(
            theta=0.5, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )
        assert len(col.species['O2']['theta']) == col.N
        assert_allclose(col.species['O2']['theta'], 0.5)

    def test_add_species_custom_advection(self):
        """Species can have custom advection velocity."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001, w=0.5)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux',
            w=0.1  # Custom advection
        )
        assert col.species['O2']['w'] == 0.1

    def test_add_species_default_advection(self):
        """Species without custom w should use column's w."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001, w=0.5)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )
        assert col.species['O2']['w'] == 0.5

    def test_add_species_explicit_zero_advection(self):
        """Explicit w=0 should override a nonzero column default."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.001, w=0.5)
        col.add_species(
            theta=1, name='O2', D=1e-5, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux',
            w=0
        )
        assert col.species['O2']['w'] == 0


class TestColumnTransport:
    """Tests for transport equation integration."""

    def test_pure_diffusion_steady_state(self):
        """Pure diffusion with constant BCs should reach linear steady state."""
        col = Column(length=1, dx=0.1, tend=10.0, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=0.1, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )

        col.solve(verbose=False)

        # Steady state for pure diffusion with Dirichlet BCs is linear
        expected = 1.0 - col.x / col.length
        actual = col.species['C']['concentration'][:, -1]
        assert_allclose(actual, expected, atol=0.01)

    def test_dirichlet_bc_preserved(self):
        """Dirichlet boundary values should be maintained throughout."""
        col = Column(length=10, dx=0.5, tend=0.5, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=0.5,
            bc_top_value=1.0, bc_top_type='dirichlet',
            bc_bot_value=0.0, bc_bot_type='dirichlet'
        )

        col.solve(verbose=False)

        # Check boundary values at all times
        top_values = col.species['C']['concentration'][0, :]
        bot_values = col.species['C']['concentration'][-1, :]
        assert_allclose(top_values, 1.0)
        assert_allclose(bot_values, 0.0)

    def test_transport_with_advection(self):
        """Transport with advection should show downstream movement."""
        col = Column(length=10, dx=0.1, tend=0.5, dt=0.001, w=1.0)
        col.add_species(
            theta=1, name='C', D=0.01, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )

        col.solve(verbose=False)

        # Concentration should decrease with depth
        final = col.species['C']['concentration'][:, -1]
        assert final[0] > final[-1]

    def test_no_transport_flag(self):
        """Species with int_transport=False should not have transport matrices."""
        col = Column(length=10, dx=1, tend=0.1, dt=0.01, w=1.0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=5.0,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux',
            int_transport=False
        )

        # With int_transport=False, no AL/AR matrices should be created
        assert 'AL' not in col.species['C']
        assert 'AR' not in col.species['C']
        assert col.species['C']['int_transport'] is False


class TestColumnAnalyticalComparison:
    """Tests comparing numerical to analytical solutions."""

    @pytest.mark.slow
    def test_advection_diffusion_analytical(self):
        """Compare to analytical solution for ADR equation."""
        w = 0.2
        D = 40
        tend = 0.1
        dx = 0.1
        length = 100
        dt = 0.001

        col = Column(length, dx, tend, dt, w)
        col.add_species(
            theta=1, name='O2', D=D, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )

        col.solve(verbose=False)

        # Analytical solution (erfc for semi-infinite domain)
        x = col.x
        sol = 0.5 * (
            special.erfc((x - w * tend) / 2 / np.sqrt(D * tend)) +
            np.exp(w * x / D) *
            special.erfc((x + w * tend) / 2 / np.sqrt(D * tend))
        )

        # Compare only in region away from bottom boundary
        actual = col.species['O2']['concentration'][:, -1]
        # Use only first half of domain to avoid boundary effects
        n_compare = len(x) // 2
        assert_allclose(actual[:n_compare], sol[:n_compare], atol=0.05)


class TestColumnReactions:
    """Tests for reactions in Column model."""

    def test_first_order_decay_in_column(self):
        """Test first-order decay with transport."""
        col = Column(length=10, dx=0.5, tend=1.0, dt=0.01, w=0, ode_method='rk4')
        col.add_species(
            theta=1, name='A', D=0.1, init_conc=1.0,
            bc_top_value=1.0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='flux'
        )
        col.constants['k'] = 1.0
        col.rates['R'] = 'k * A'
        col.dcdt['A'] = '-R'

        col.solve(verbose=False)

        # All concentrations should decrease due to decay
        init = col.species['A']['concentration'][:, 0]
        final = col.species['A']['concentration'][:, -1]
        # Interior points (not BC) should decay
        assert np.all(final[1:] <= init[1:])


class TestColumnBoundaryConditions:
    """Tests for different boundary condition combinations."""

    def test_neumann_neumann_bc(self):
        """Test flux BCs at both boundaries."""
        col = Column(length=10, dx=0.5, tend=0.1, dt=0.001, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=1.0,
            bc_top_value=0, bc_top_type='flux',
            bc_bot_value=0, bc_bot_type='flux'
        )

        col.solve(verbose=False)

        # With zero flux at both ends, total mass should be conserved
        init_mass = col.species['C']['concentration'][:, 0].sum()
        final_mass = col.species['C']['concentration'][:, -1].sum()
        assert_allclose(final_mass, init_mass, rtol=0.01)

    def test_mixed_bc_neumann_dirichlet(self):
        """Test flux at top, constant at bottom."""
        col = Column(length=10, dx=0.5, tend=0.5, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=0.5,
            bc_top_value=0.1, bc_top_type='flux',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )

        col.solve(verbose=False)

        # Bottom BC should be maintained
        assert_allclose(col.species['C']['concentration'][-1, -1], 0.0)


class TestColumnChangeBoundaryConditions:
    """Tests for dynamic boundary condition changes."""

    def test_change_bc_updates_species(self):
        """change_boundary_conditions should update species BC values."""
        col = Column(length=10, dx=1, tend=0.1, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )

        col.change_boundary_conditions(
            'C', i=5,
            bc_top_value=2.0, bc_top_type='dirichlet',
            bc_bot_value=0.5, bc_bot_type='flux'
        )

        assert col.species['C']['bc_top_value'] == 2.0
        assert col.species['C']['bc_bot_type'] == 'flux'


class TestColumnFluxEstimation:
    """Tests for flux estimation methods."""

    def test_estimate_flux_at_top(self):
        """estimate_flux_at_top should return flux array."""
        col = Column(length=10, dx=0.5, tend=0.1, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )
        col.solve(verbose=False)

        flux = col.estimate_flux_at_top('C', order=2)
        assert len(flux) == len(col.time)

    def test_estimate_flux_at_bottom(self):
        """estimate_flux_at_bottom should return flux array."""
        col = Column(length=10, dx=0.5, tend=0.1, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )
        col.solve(verbose=False)

        flux = col.estimate_flux_at_bottom('C', order=2)
        assert len(flux) == len(col.time)

    def test_flux_estimators_apply_idx_to_advective_term(self):
        """Flux estimators should slice advective concentrations consistently."""
        col = Column(length=10, dx=1, tend=0.2, dt=0.1, w=0)
        col.add_species(
            theta=1, name='C', D=0, init_conc=0,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet',
            w=2,
            int_transport=False
        )
        col.species['C']['concentration'][0, :] = [1.0, 2.0, 3.0]
        col.species['C']['concentration'][-1, :] = [4.0, 5.0, 6.0]

        idx = slice(1, None)
        assert_allclose(col.estimate_flux_at_top('C', idx=idx, order=1),
                        [-4.0, -6.0])
        assert_allclose(col.estimate_flux_at_bottom('C', idx=idx, order=1),
                        [10.0, 12.0])

    def test_flux_estimators_reject_invalid_order(self):
        """Only implemented finite-difference orders should be accepted."""
        col = Column(length=10, dx=1, tend=0.1, dt=0.1, w=0)
        col.add_species(
            theta=1, name='C', D=0, init_conc=0,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet',
            int_transport=False
        )
        with pytest.raises(ValueError, match="order must be"):
            col.estimate_flux_at_top('C', order=5)

    def test_flux_estimators_match_prerefactor_snapshot(self):
        """Lock exact finite-difference flux outputs (orders 1-4, top & bottom)
        on a fixture with D!=0, w!=0 and non-uniform theta, so the flux
        de-duplication cannot change numerical behavior. Snapshot captured from
        the pre-refactor implementation."""
        col = Column(length=6, dx=1.0, tend=0.2, dt=0.1, w=0.5)
        col.add_species(
            theta=1, name='C', D=2.0, init_conc=0,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet',
            w=0.5, int_transport=False
        )
        col.species['C']['theta'] = np.array(
            [1.0, 0.9, 0.8, 0.85, 0.95, 0.7, 0.75])
        col.species['C']['concentration'] = (
            (np.arange(7)[:, None] + 1.0) + np.array([0.0, 0.5, 1.0])[None, :])

        expected_top = {
            1: [0.9000000000000004, 0.5500000000000003, 0.20000000000000018],
            2: [0.30000000000000115, -0.12499999999999867, -0.5500000000000007],
            3: [0.26666666666666805, -0.19166666666666643, -0.650000000000001],
            4: [1.3666666666666714, 0.9833333333333403, 0.6000000000000023],
        }
        expected_bottom = {
            1: [2.125, 2.4124999999999996, 2.6999999999999993],
            2: [5.625000000000002, 6.237500000000001, 6.85],
            3: [7.124999999999997, 7.87083333333333, 8.616666666666667],
            4: [8.224999999999998, 9.045833333333327, 9.86666666666666],
        }
        idx = slice(1, None)
        for order in (1, 2, 3, 4):
            assert_allclose(col.estimate_flux_at_top('C', order=order),
                            expected_top[order])
            assert_allclose(col.estimate_flux_at_bottom('C', order=order),
                            expected_bottom[order])
            # idx slicing must select the matching time columns.
            assert_allclose(col.estimate_flux_at_top('C', idx=idx, order=order),
                            expected_top[order][1:])
            assert_allclose(
                col.estimate_flux_at_bottom('C', idx=idx, order=order),
                expected_bottom[order][1:])

    def test_flux_estimators_match_prerefactor_snapshot_non_unit_dx(self):
        """Same flux characterization at dx=0.5, so the /self.dx factor is
        actually exercised (the dx=1.0 snapshot cannot catch a dropped or
        misplaced dx). Snapshot captured from the refactored implementation."""
        col = Column(length=3, dx=0.5, tend=0.2, dt=0.1, w=0.5)
        col.add_species(
            theta=1, name='C', D=2.0, init_conc=0,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet',
            w=0.5, int_transport=False
        )
        col.species['C']['theta'] = np.array(
            [1.0, 0.9, 0.8, 0.85, 0.95, 0.7, 0.75])
        col.species['C']['concentration'] = (
            (np.arange(7)[:, None] + 1.0) + np.array([0.0, 0.5, 1.0])[None, :])

        expected_top = {
            1: [2.3000000000000007, 1.8500000000000005, 1.4000000000000004],
            2: [1.1000000000000023, 0.5000000000000027, -0.10000000000000142],
            3: [1.033333333333336, 0.36666666666666714, -0.30000000000000193],
            4: [3.2333333333333427, 2.7166666666666806, 2.2000000000000046],
        }
        expected_bottom = {
            1: [1.625, 2.0124999999999993, 2.3999999999999986],
            2: [8.625000000000004, 9.662500000000001, 10.7],
            3: [11.624999999999995, 12.92916666666666, 14.233333333333334],
            4: [13.824999999999996, 15.279166666666656, 16.73333333333332],
        }
        for order in (1, 2, 3, 4):
            assert_allclose(col.estimate_flux_at_top('C', order=order),
                            expected_top[order])
            assert_allclose(col.estimate_flux_at_bottom('C', order=order),
                            expected_bottom[order])


class TestColumnAcidBase:
    """Tests for acid-base equilibrium concentration updates in Column."""

    def test_acid_base_update_concentrations_redistributes_by_alpha(self):
        """Column.acid_base_update_concentrations should redistribute each
        component's total across its species by the per-depth speciation
        fractions (2D alpha path), leave neutral ions unchanged, and NOT store
        an alpha field (Column species have none). Snapshot from the
        pre-refactor implementation (N>1, varying pH profile)."""
        col = Column(length=2, dx=1, tend=0.02, dt=0.01, w=0)
        common = dict(bc_top_value=0, bc_top_type='dirichlet',
                      bc_bot_value=0, bc_bot_type='dirichlet',
                      int_transport=False)
        col.add_species(theta=1, name='HAc', D=0, init_conc=0.1, **common)
        col.add_species(theta=1, name='Ac', D=0, init_conc=0.0, **common)
        col.add_acid(['HAc', 'Ac'], pKa=[4.76], charge=0)
        col.add_species(theta=1, name='Na', D=0, init_conc=0.05, **common)
        col.add_ion(name='Na', charge=1)
        col.create_acid_base_system()

        i = 1
        col.species['pH']['concentration'][:, i] = [4.0, 4.76, 5.5]
        col.species['HAc']['concentration'][:, i] = [0.08, 0.05, 0.03]
        col.species['Ac']['concentration'][:, i] = [0.02, 0.05, 0.07]
        col.species['Na']['concentration'][:, i] = [0.05, 0.05, 0.05]
        col.acid_base_update_concentrations(i)

        assert_allclose(
            col.species['HAc']['concentration'][:, i],
            [0.08519483458525738, 0.05, 0.015395489956790518])
        assert_allclose(
            col.species['Ac']['concentration'][:, i],
            [0.014805165414742631, 0.05, 0.08460451004320949])
        assert_allclose(col.species['Na']['concentration'][:, i],
                        [0.05, 0.05, 0.05])
        # total acid concentration is conserved per depth
        assert_allclose(
            col.species['HAc']['concentration'][:, i]
            + col.species['Ac']['concentration'][:, i], [0.1, 0.1, 0.1])
        assert 'alpha' not in col.species['HAc']

    def test_solve_end_to_end_acid_base_speciation(self):
        """End-to-end: Column.solve() (N>1, with transport) drives the real
        dispatch onto the inherited acid_base_update_concentrations and yields
        pH-consistent per-depth speciation with a conserved total. Guards the
        unified base method through solve() for the spatial case."""
        col = Column(length=2, dx=1, tend=0.03, dt=0.01, w=0)

        def add_acid_species(name, init):
            # int_transport=True so solve() propagates concentrations; Dirichlet
            # BCs equal to init keep the profile steady for transport.
            col.add_species(theta=1, name=name, D=0.01, init_conc=init,
                            bc_top_value=init, bc_top_type='dirichlet',
                            bc_bot_value=init, bc_bot_type='dirichlet',
                            int_transport=True)

        add_acid_species('HAc', 0.08)
        add_acid_species('Ac', 0.02)
        col.add_acid(['HAc', 'Ac'], pKa=[4.76], charge=0)
        add_acid_species('Na', 0.05)
        col.add_ion(name='Na', charge=1)
        col.create_acid_base_system()
        col.solve(verbose=False)

        pH = col.species['pH']['concentration'][:, -1]
        HAc = col.species['HAc']['concentration'][:, -1]
        Ac = col.species['Ac']['concentration'][:, -1]
        # pH solved at every depth (off the default 7.0)
        assert np.all(pH < 6.0)
        # total conserved per depth
        assert_allclose(HAc + Ac, 0.10, rtol=1e-6)
        # speciation consistent with the solved pH per depth (Henderson-Hasselbalch)
        ratio = 10.0 ** (pH - 4.76)
        assert_allclose(HAc, (HAc + Ac) / (1.0 + ratio), rtol=1e-6)
        # Column does not store alpha
        assert 'alpha' not in col.species['HAc']


class TestColumnPlotMethods:
    """Tests that plot methods exist (smoke tests)."""

    def test_has_plot_methods(self):
        """Column should have plot methods defined."""
        col = Column(length=10, dx=1, tend=1.0, dt=0.01)
        assert hasattr(col, 'plot_profiles')
        assert hasattr(col, 'plot_profile')
        assert hasattr(col, 'contour_plot')
        assert hasattr(col, 'plot_depths')
        assert hasattr(col, 'plot_times')


class TestColumnSaveLoad:
    """Tests for saving and loading profiles."""

    def test_save_final_profiles(self, tmp_path):
        """save_final_profiles should create CSV files."""
        col = Column(length=10, dx=1, tend=0.01, dt=0.001, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=1.0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )
        col.solve(verbose=False)
        col.save_final_profiles(directory=tmp_path)

        assert (tmp_path / 'C.csv').exists()

    def test_load_initial_conditions_from_directory(self, tmp_path):
        """load_initial_conditions should read profiles from the requested directory."""
        source = Column(length=10, dx=1, tend=0.01, dt=0.001, w=0)
        source.add_species(
            theta=1, name='C', D=1.0, init_conc=np.linspace(0, 1, 11),
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=1, bc_bot_type='dirichlet'
        )
        source.save_final_profiles(directory=tmp_path)

        target = Column(length=10, dx=1, tend=0.01, dt=0.001, w=0)
        target.add_species(
            theta=1, name='C', D=1.0, init_conc=0,
            bc_top_value=0, bc_top_type='dirichlet',
            bc_bot_value=1, bc_bot_type='dirichlet'
        )
        target.load_initial_conditions(directory=tmp_path)

        assert_allclose(target.species['C']['init_conc'], source.profiles['C'])
