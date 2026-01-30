"""Smoke tests for plotting functions.

These tests verify that plotting functions execute without errors.
They don't validate the visual output - just that the code runs.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from porousmedialab.batch import Batch
from porousmedialab.column import Column
from porousmedialab.phcalc import Acid


class TestBatchPlotSmoke:
    """Smoke tests for Batch plotting methods."""

    @pytest.fixture
    def solved_batch(self):
        """Create a solved batch for plotting tests."""
        batch = Batch(tend=1.0, dt=0.01)
        batch.add_species('A', init_conc=1.0)
        batch.add_species('B', init_conc=0.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'
        batch.dcdt['B'] = 'R'
        batch.solve(verbose=False)
        batch.reconstruct_rates()
        return batch

    def test_plot_runs(self, solved_batch):
        """plot method should execute without error."""
        try:
            ax = solved_batch.plot('A')
            assert ax is not None
        finally:
            plt.close('all')

    def test_plot_profiles_runs(self, solved_batch):
        """plot_profiles should execute without error."""
        try:
            solved_batch.plot_profiles()
        finally:
            plt.close('all')


class TestColumnPlotSmoke:
    """Smoke tests for Column plotting methods."""

    @pytest.fixture
    def solved_column(self):
        """Create a solved column for plotting tests."""
        col = Column(length=10, dx=0.5, tend=0.5, dt=0.01, w=0)
        col.add_species(
            theta=1, name='C', D=1.0, init_conc=0,
            bc_top_value=1, bc_top_type='dirichlet',
            bc_bot_value=0, bc_bot_type='dirichlet'
        )
        col.solve(verbose=False)
        return col

    def test_plot_profile_runs(self, solved_column):
        """plot_profile should execute without error."""
        try:
            ax = solved_column.plot_profile('C')
            assert ax is not None
        finally:
            plt.close('all')

    def test_plot_profiles_runs(self, solved_column):
        """plot_profiles should execute without error."""
        try:
            solved_column.plot_profiles()
        finally:
            plt.close('all')

    def test_contour_plot_runs(self, solved_column):
        """contour_plot should execute without error."""
        try:
            ax = solved_column.contour_plot('C')
            assert ax is not None
        finally:
            plt.close('all')

    def test_plot_depths_runs(self, solved_column):
        """plot_depths should execute without error."""
        try:
            ax = solved_column.plot_depths('C', depths=[0, 2, 4])
            assert ax is not None
        finally:
            plt.close('all')

    def test_plot_times_runs(self, solved_column):
        """plot_times should execute without error."""
        try:
            ax = solved_column.plot_times('C', time_slices=[0.1, 0.3, 0.5])
            assert ax is not None
        finally:
            plt.close('all')


class TestCustomPlotSmoke:
    """Smoke tests for custom plot function."""

    def test_custom_plot_runs(self):
        """custom_plot should execute without error."""
        from porousmedialab.plotter import custom_plot

        col = Column(length=10, dx=1, tend=1.0, dt=0.01)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        try:
            ax = custom_plot(col, x, y, ttl='Test', y_lbl='Y', x_lbl='X')
            assert ax is not None
        finally:
            plt.close('all')


class TestFigureCleanup:
    """Tests that figures are properly created."""

    def test_figures_created(self):
        """Verify that plotting creates figure objects."""
        from porousmedialab.plotter import custom_plot

        col = Column(length=10, dx=1, tend=1.0, dt=0.01)
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])

        try:
            # Record figure count before
            n_before = len(plt.get_fignums())

            custom_plot(col, x, y)

            # Should have created at least one figure
            n_after = len(plt.get_fignums())
            assert n_after > n_before
        finally:
            plt.close('all')
