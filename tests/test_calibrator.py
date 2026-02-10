"""Tests for Calibrator module (parameter optimization)."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from collections import OrderedDict

from porousmedialab.calibrator import Calibrator, find_indexes_of_intersections
from porousmedialab.batch import Batch


class TestFindIndexesOfIntersections:
    """Tests for the intersection finding utility."""

    def test_exact_matches(self):
        """Should find exact matches."""
        simulated = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        observed = np.array([0.1, 0.3, 0.5])
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert len(idxs) == 3
        assert 1 in idxs  # 0.1
        assert 3 in idxs  # 0.3
        assert 5 in idxs  # 0.5

    def test_approximate_matches(self):
        """Should find approximate matches within epsilon."""
        simulated = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        observed = np.array([0.105])  # Close to 0.1
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert len(idxs) == 1
        assert 1 in idxs

    def test_no_matches(self):
        """Should return empty array if no matches."""
        simulated = np.array([0.0, 0.1, 0.2])
        observed = np.array([0.5, 0.6])  # No overlap
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert len(idxs) == 0

    def test_first_match_used(self):
        """When multiple matches, first should be used."""
        simulated = np.array([0.0, 0.1, 0.1001, 0.2])  # Two values near 0.1
        observed = np.array([0.1])
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert len(idxs) == 1
        assert idxs[0] == 1  # First match

    def test_large_input(self):
        """Should work correctly with larger arrays."""
        simulated = np.linspace(0, 10, 1001)
        observed = np.array([1.0, 5.0, 9.0])
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert len(idxs) == 3
        # Verify values at found indexes are close to observed
        for i, o_val in enumerate(observed):
            assert abs(simulated[idxs[i]] - o_val) <= eps * 1.01

    def test_returns_int_array(self):
        """Return type should be numpy int array."""
        simulated = np.array([0.0, 0.1, 0.2])
        observed = np.array([0.1])
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert isinstance(idxs, np.ndarray)
        assert idxs.dtype == int

    def test_empty_observed_returns_empty_int_array(self):
        """Empty observed array should return empty int array."""
        simulated = np.array([0.0, 0.1, 0.2])
        observed = np.array([])
        eps = 0.01

        idxs = find_indexes_of_intersections(simulated, observed, eps)

        assert isinstance(idxs, np.ndarray)
        assert idxs.dtype == int
        assert len(idxs) == 0


class TestCalibratorInitialization:
    """Tests for Calibrator initialization."""

    def test_init_stores_lab(self, simple_batch):
        """Calibrator should store reference to lab."""
        cal = Calibrator(simple_batch)
        assert cal.lab is simple_batch

    def test_init_creates_empty_parameters(self, simple_batch):
        """Calibrator should start with empty parameters."""
        cal = Calibrator(simple_batch)
        assert isinstance(cal.parameters, OrderedDict)
        assert len(cal.parameters) == 0

    def test_init_creates_empty_measurements(self, simple_batch):
        """Calibrator should start with empty measurements."""
        cal = Calibrator(simple_batch)
        assert len(cal.measurements) == 0

    def test_init_error_is_nan(self, simple_batch):
        """Initial error should be NaN."""
        cal = Calibrator(simple_batch)
        assert np.isnan(cal.error)


class TestCalibratorAddParameter:
    """Tests for adding parameters to calibrate."""

    def test_add_parameter_stores_bounds(self, simple_batch):
        """add_parameter should store boundaries."""
        simple_batch.constants['k'] = 1.0
        cal = Calibrator(simple_batch)

        cal.add_parameter('k', lower_boundary=0.1, upper_boundary=10.0)

        assert 'k' in cal.parameters
        assert cal.parameters['k']['lower_boundary'] == 0.1
        assert cal.parameters['k']['upper_boundary'] == 10.0

    def test_add_parameter_stores_initial_value(self, simple_batch):
        """add_parameter should store current value from model."""
        simple_batch.constants['k'] = 2.5
        cal = Calibrator(simple_batch)

        cal.add_parameter('k', lower_boundary=0.1, upper_boundary=10.0)

        assert cal.parameters['k']['value'] == 2.5

    def test_add_multiple_parameters(self, simple_batch):
        """Should be able to add multiple parameters."""
        simple_batch.constants['k1'] = 1.0
        simple_batch.constants['k2'] = 2.0
        cal = Calibrator(simple_batch)

        cal.add_parameter('k1', 0.1, 10.0)
        cal.add_parameter('k2', 0.5, 5.0)

        assert len(cal.parameters) == 2


class TestCalibratorAddMeasurement:
    """Tests for adding measurements."""

    def test_add_measurement_stores_values(self, simple_batch):
        """add_measurement should store measurement data."""
        cal = Calibrator(simple_batch)
        values = np.array([0.9, 0.8, 0.7])
        time = np.array([0.1, 0.2, 0.3])

        cal.add_measurement('A', values, time)

        assert 'A' in cal.measurements
        assert_allclose(cal.measurements['A']['values'], values)
        assert_allclose(cal.measurements['A']['time'], time)

    def test_add_measurement_default_depth(self, simple_batch):
        """Default depth should be 0."""
        cal = Calibrator(simple_batch)
        cal.add_measurement('A', np.array([1.0]), np.array([0.1]))

        assert cal.measurements['A']['depth'] == 0

    def test_add_measurement_custom_depth(self, simple_batch):
        """Should accept custom depth."""
        cal = Calibrator(simple_batch)
        cal.add_measurement('A', np.array([1.0]), np.array([0.1]), depth=5)

        assert cal.measurements['A']['depth'] == 5


class TestCalibratorIterParams:
    """Tests for parameter iteration."""

    def test_iter_params_returns_x0_and_bounds(self, simple_batch):
        """iter_params should return initial values and bounds."""
        simple_batch.constants['k'] = 1.5
        cal = Calibrator(simple_batch)
        cal.add_parameter('k', 0.1, 10.0)

        x0, bounds = cal.iter_params()

        assert x0 == [1.5]
        assert bounds == [(0.1, 10.0)]

    def test_iter_params_preserves_order(self, simple_batch):
        """iter_params should preserve parameter order."""
        simple_batch.constants['a'] = 1.0
        simple_batch.constants['b'] = 2.0
        simple_batch.constants['c'] = 3.0
        cal = Calibrator(simple_batch)
        cal.add_parameter('a', 0, 5)
        cal.add_parameter('b', 0, 5)
        cal.add_parameter('c', 0, 5)

        x0, bounds = cal.iter_params()

        assert x0 == [1.0, 2.0, 3.0]


class TestCalibratorMinFunction:
    """Tests for the minimization function."""

    def test_min_function_updates_constants(self):
        """min_function should update model constants."""
        from porousmedialab.metrics import rmse

        batch = Batch(tend=0.1, dt=0.01)
        batch.add_species('A', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'

        cal = Calibrator(batch)
        cal.add_parameter('k', 0.1, 10.0)
        # Use multiple measurement points with variance for norm_rmse compatibility
        cal.add_measurement('A', np.array([0.98, 0.96, 0.94]), np.array([0.02, 0.04, 0.06]))

        # Call min_function with new k value
        cal.min_function([2.0])

        assert batch.constants['k'] == 2.0

    def test_min_function_returns_error(self):
        """min_function should return error value."""
        batch = Batch(tend=0.1, dt=0.01)
        batch.add_species('A', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'

        cal = Calibrator(batch)
        cal.add_parameter('k', 0.1, 10.0)
        # Use multiple measurement points with variance for norm_rmse compatibility
        cal.add_measurement('A', np.array([0.98, 0.96, 0.94]), np.array([0.02, 0.04, 0.06]))

        error = cal.min_function([1.0])

        assert isinstance(error, float)
        assert not np.isnan(error)


class TestCalibratorEstimateError:
    """Tests for error estimation."""

    def test_estimate_error_small_for_good_match(self):
        """Error should be small when simulation matches measurement."""
        from porousmedialab.metrics import rmse  # Use RMSE instead of norm_rmse

        batch = Batch(tend=0.1, dt=0.01)
        batch.add_species('A', init_conc=1.0)
        batch.constants['k'] = 1.0
        batch.rates['R'] = 'k * A'
        batch.dcdt['A'] = '-R'
        batch.solve(verbose=False)

        # Use actual model output as "measurement" at multiple times
        times = np.array([0.02, 0.04, 0.06, 0.08])
        idxs = (times / batch.dt).astype(int)
        measured = batch.species['A']['concentration'][0, idxs]

        cal = Calibrator(batch)
        cal.add_measurement('A', measured, times)
        # Use RMSE instead of norm_rmse to avoid divide by zero
        cal.estimate_error(metric_fun=rmse, disp=False)

        assert cal.error < 1e-6


class TestCalibratorIntegration:
    """Integration tests for full calibration workflow."""

    @pytest.mark.slow
    def test_calibrate_simple_decay(self):
        """Test calibration of decay rate constant."""
        # True model with k=2.0
        true_k = 2.0
        batch_true = Batch(tend=0.5, dt=0.01)
        batch_true.add_species('A', init_conc=1.0)
        batch_true.constants['k'] = true_k
        batch_true.rates['R'] = 'k * A'
        batch_true.dcdt['A'] = '-R'
        batch_true.solve(verbose=False)

        # Generate "measurements" at specific times
        meas_times = np.array([0.1, 0.2, 0.3, 0.4])
        meas_idxs = (meas_times / batch_true.dt).astype(int)
        meas_values = batch_true.species['A']['concentration'][0, meas_idxs]

        # Model to calibrate with wrong initial guess
        batch_cal = Batch(tend=0.5, dt=0.01)
        batch_cal.add_species('A', init_conc=1.0)
        batch_cal.constants['k'] = 0.5  # Wrong initial guess
        batch_cal.rates['R'] = 'k * A'
        batch_cal.dcdt['A'] = '-R'

        cal = Calibrator(batch_cal)
        cal.add_parameter('k', 0.1, 10.0)
        cal.add_measurement('A', meas_values, meas_times)

        # Run calibration
        cal.run(verbose=False)

        # Calibrated k should be close to true k
        assert_allclose(batch_cal.constants['k'], true_k, rtol=0.1)
