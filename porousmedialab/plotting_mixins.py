"""Plotting alias mixins for the simulation classes.

These mixins keep visualization wiring out of the simulation classes (``Batch``,
``Column``) while preserving the exact public plotting API. Each plotting method
is a class attribute bound to a function in :mod:`porousmedialab.plotter`; the
functions take the lab/column instance as their first positional argument, so
``instance.plot_x(...)`` binds ``self`` as usual.

``plot_profiles`` deliberately maps to different functions for the two models
(``all_plot_depth_index`` for batch index plots vs ``plot_profiles`` for column
depth profiles), so the mappings are kept on separate mixins rather than shared.
"""

import porousmedialab.plotter as plotter


class BatchPlottingMixin:
    """Plotting methods for the 0-D :class:`~porousmedialab.batch.Batch` model."""

    plot = plotter.plot_depth_index
    plot_profiles = plotter.all_plot_depth_index
    plot_fractions = plotter.plot_fractions
    plot_rates = plotter.plot_batch_rates
    plot_rate = plotter.plot_batch_rate
    plot_deltas = plotter.plot_batch_deltas
    plot_delta = plotter.plot_batch_delta


class ColumnPlottingMixin:
    """Plotting methods for the 1-D :class:`~porousmedialab.column.Column` model."""

    custom_plot = plotter.custom_plot
    plot_depths = plotter.plot_depths
    plot_times = plotter.plot_times
    plot_profiles = plotter.plot_profiles
    plot_profile = plotter.plot_profile
    plot_contourplots = plotter.plot_contourplots
    contour_plot = plotter.contour_plot
    plot_contourplots_of_rates = plotter.plot_contourplots_of_rates
    contour_plot_of_rates = plotter.contour_plot_of_rates
    plot_contourplots_of_deltas = plotter.plot_contourplots_of_deltas
    contour_plot_of_delta = plotter.contour_plot_of_delta
    plot_saturation_index = plotter.saturation_index_countour
