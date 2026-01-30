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
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            col = Column(length=10, dx=1, tend=0.01, dt=0.001, w=0)
            col.add_species(
                theta=1, name='C', D=1.0, init_conc=1.0,
                bc_top_value=1, bc_top_type='dirichlet',
                bc_bot_value=0, bc_bot_type='dirichlet'
            )
            col.solve(verbose=False)
            col.save_final_profiles()

            assert (tmp_path / 'C.csv').exists()
        finally:
            os.chdir(original_cwd)
