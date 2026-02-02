"""Tests for black-box optimization module."""

import pytest
import numpy as np
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from porousmedialab.blackbox import latin, rbf, search, get_default_executor


# Module-level functions that can be pickled for multiprocessing
def sphere_func(x):
    """Sphere function - minimum at origin."""
    return sum(xi**2 for xi in x)


def sphere_shifted_func(x):
    """Sphere function shifted - minimum at (0.5, 0.5)."""
    return sum((xi - 0.5)**2 for xi in x)


def sum_func(x):
    """Simple sum function."""
    return sum(x)


def noisy_sphere_func(x):
    """Sphere function with small noise."""
    return sum(xi**2 for xi in x) + np.random.rand() * 0.01


def parabola_1d_func(x):
    """1D parabola - minimum at 0.3."""
    return (x[0] - 0.3)**2


def identity_func(x):
    """Return first coordinate."""
    return x[0]


def rosenbrock_scaled_func(x):
    """Rosenbrock function scaled to [0, 1] box."""
    a, b = 1, 100
    x1, x2 = x[0] * 2, x[1] * 2  # Scale back to [0, 2]
    return (a - x1)**2 + b * (x2 - x1**2)**2


def double_func(x):
    """Double the input."""
    return x * 2


# Mock executor that uses sequential execution (no multiprocessing)
# This avoids pickling issues in tests
@contextmanager
def mock_executor():
    """A simple executor that runs functions sequentially."""
    class MockPool:
        def map(self, func, iterable):
            return [func(x) for x in iterable]
    yield MockPool()


def get_mock_executor():
    """Return the mock executor factory."""
    return mock_executor


class TestLatin:
    """Tests for Latin hypercube sampling."""

    def test_latin_returns_correct_shape(self):
        """Latin hypercube should return n x d array."""
        n, d = 10, 3
        result = latin(n, d)
        result = np.array(result)  # latin may return list
        assert result.shape == (n, d)

    def test_latin_values_in_unit_cube(self):
        """All values should be in [0, 1]."""
        result = np.array(latin(10, 3))
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_latin_covers_space(self):
        """Each dimension should have values spread across [0, 1]."""
        n, d = 20, 2
        result = np.array(latin(n, d))
        # Check that min and max are reasonably spread
        for dim in range(d):
            assert np.min(result[:, dim]) < 0.2
            assert np.max(result[:, dim]) > 0.8

    def test_latin_no_duplicate_rows(self):
        """Points should be distinct."""
        result = np.array(latin(10, 2))
        # Each row should be unique
        unique_rows = np.unique(result, axis=0)
        assert len(unique_rows) == len(result)

    def test_latin_minimum_points(self):
        """Latin hypercube with minimum points should work."""
        # n=2 is the minimum for the algorithm to work due to division by n-1
        result = np.array(latin(2, 3))
        assert result.shape == (2, 3)

    def test_latin_high_dimension(self):
        """Latin hypercube should work with higher dimensions."""
        n, d = 5, 10
        result = np.array(latin(n, d))
        assert result.shape == (n, d)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_latin_values_are_evenly_distributed(self):
        """Values in each dimension should be from a regular grid."""
        n, d = 10, 2
        result = np.array(latin(n, d))
        # Each dimension should have values at i/(n-1) for i in 0..n-1
        expected_values = set(i / (n - 1) for i in range(n))
        for dim in range(d):
            actual_values = set(result[:, dim])
            assert actual_values == expected_values


class TestRBF:
    """Tests for Radial Basis Function interpolation."""

    def test_rbf_returns_callable(self):
        """RBF should return a callable function."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 4.0],
        ])
        T = np.identity(2)
        fit = rbf(points, T)
        assert callable(fit)

    def test_rbf_interpolates_known_points(self):
        """RBF should pass through the known points."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 4.0],
        ])
        T = np.identity(2)
        fit = rbf(points, T)

        # Check interpolation at known points
        for i in range(len(points)):
            x = points[i, :-1]
            expected = points[i, -1]
            result = fit(x)
            assert np.isclose(result, expected, rtol=1e-5)

    def test_rbf_returns_numeric_value(self):
        """RBF fit should return numeric values for new points."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 4.0],
        ])
        T = np.identity(2)
        fit = rbf(points, T)

        # Test at new point
        result = fit([0.5, 0.5])
        assert isinstance(result, (int, float, np.number))
        assert not np.isnan(result)

    def test_rbf_with_scaling_matrix(self):
        """RBF should work with non-identity scaling matrix."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 4.0],
        ])
        # Non-identity scaling matrix
        T = np.array([[2.0, 0.0], [0.0, 0.5]])
        fit = rbf(points, T)

        # Should still interpolate known points
        for i in range(len(points)):
            x = points[i, :-1]
            expected = points[i, -1]
            result = fit(x)
            assert np.isclose(result, expected, rtol=1e-5)

    def test_rbf_1d(self):
        """RBF should work with 1D input."""
        points = np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 0.0],
        ])
        T = np.identity(1)
        fit = rbf(points, T)

        # Check interpolation at known points
        for i in range(len(points)):
            x = points[i, :-1]
            expected = points[i, -1]
            result = fit(x)
            assert np.isclose(result, expected, rtol=1e-5)

    def test_rbf_smooth_interpolation(self):
        """RBF should produce smooth interpolation between points."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ])
        T = np.identity(2)
        fit = rbf(points, T)

        # Value at center should be reasonable interpolation
        center_value = fit([0.5, 0.5])
        # Should be between min and max of known values
        assert 0.0 <= center_value <= 2.0


class TestSearch:
    """Tests for the main search/optimization function.

    Uses a mock executor to avoid multiprocessing pickling issues.
    """

    def test_search_creates_output_file(self):
        """Search should create a results file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=sphere_func,
                box=[[0, 1], [0, 1]],
                n=4,
                m=4,
                batch=2,
                resfile=resfile,
                executor=get_mock_executor()
            )
            assert os.path.exists(resfile)
            # Check file has content
            with open(resfile, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1  # Header + data
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_finds_minimum_region(self):
        """Search should find points near the minimum."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=sphere_shifted_func,
                box=[[0, 1], [0, 1]],
                n=8,
                m=8,
                batch=4,
                resfile=resfile,
                executor=get_mock_executor()
            )

            # Load results and check best point is near minimum
            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            best_value = data[0, -1]

            # Best point should be reasonably close to minimum
            assert best_value < 0.1  # Should find something better than random
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_output_format(self):
        """Search output file should have correct format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=sum_func,
                box=[[0, 1], [0, 1]],
                n=4,
                m=4,
                batch=2,
                resfile=resfile,
                executor=get_mock_executor()
            )

            with open(resfile, 'r') as f:
                header = f.readline()
                # Header should contain parameter labels and f_value
                assert 'par_1' in header
                assert 'par_2' in header
                assert 'f_value' in header

            # Load and verify data shape
            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            assert data.shape[1] == 3  # 2 params + 1 f_value
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_results_sorted(self):
        """Search results should be sorted by f_value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=noisy_sphere_func,
                box=[[0, 1], [0, 1]],
                n=8,
                m=4,
                batch=4,
                resfile=resfile,
                executor=get_mock_executor()
            )

            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            f_values = data[:, -1]

            # Values should be sorted in ascending order
            assert np.all(f_values[:-1] <= f_values[1:])
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_batch_adjustment(self):
        """Search should adjust n and m to be divisible by batch."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            # n=5 should be adjusted to 6 (next multiple of 3)
            # m=7 should be adjusted to 9 (next multiple of 3)
            search(
                f=sum_func,
                box=[[0, 1]],
                n=5,
                m=7,
                batch=3,
                resfile=resfile,
                executor=get_mock_executor()
            )

            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            # Total points should be adjusted n + adjusted m = 6 + 9 = 15
            assert len(data) == 15
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_1d_optimization(self):
        """Search should work with 1D optimization problems."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=parabola_1d_func,
                box=[[0, 1]],
                n=6,
                m=6,
                batch=3,
                resfile=resfile,
                executor=get_mock_executor()
            )

            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            best_x = data[0, 0]
            best_f = data[0, 1]

            # Best point should be near 0.3
            assert abs(best_x - 0.3) < 0.3
            assert best_f < 0.1
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_custom_box(self):
        """Search should respect custom box bounds."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=identity_func,
                box=[[10, 20]],  # Non-unit box
                n=4,
                m=4,
                batch=2,
                resfile=resfile,
                executor=get_mock_executor()
            )

            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            # All x values should be within [10, 20]
            assert np.all(data[:, 0] >= 10)
            assert np.all(data[:, 0] <= 20)
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_rho0_parameter(self):
        """Search should accept rho0 parameter for balls density."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=sphere_func,
                box=[[0, 1], [0, 1]],
                n=4,
                m=4,
                batch=2,
                resfile=resfile,
                rho0=0.3,  # Custom rho0
                executor=get_mock_executor()
            )
            assert os.path.exists(resfile)
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_search_p_parameter(self):
        """Search should accept p parameter for decay rate."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=sphere_func,
                box=[[0, 1], [0, 1]],
                n=4,
                m=4,
                batch=2,
                resfile=resfile,
                p=2.0,  # Faster decay
                executor=get_mock_executor()
            )
            assert os.path.exists(resfile)
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)


class TestGetDefaultExecutor:
    """Tests for executor factory."""

    def test_returns_pool_type(self):
        """Should return a Pool-like class."""
        executor = get_default_executor()
        assert callable(executor)

    def test_executor_has_context_manager(self):
        """Executor should work as context manager."""
        executor = get_default_executor()
        # Should not raise when used as context manager
        with executor() as pool:
            assert hasattr(pool, 'map')

    def test_mock_executor_map_works(self):
        """Mock executor's map method should work correctly."""
        # Use mock executor to avoid pickling issues
        with mock_executor() as pool:
            result = list(pool.map(lambda x: x * 2, [1, 2, 3]))
            assert result == [2, 4, 6]


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_latin_rbf_integration(self):
        """Latin hypercube points should work with RBF."""
        n, d = 5, 2

        # Generate latin hypercube
        lh = np.array(latin(n, d))

        # Add function values (simple quadratic)
        points = np.zeros((n, d + 1))
        points[:, :-1] = lh
        points[:, -1] = [sum(x**2) for x in lh]

        # Build RBF
        T = np.identity(d)
        fit = rbf(points, T)

        # Should interpolate original points
        for i in range(n):
            x = points[i, :-1]
            expected = points[i, -1]
            result = fit(x)
            assert np.isclose(result, expected, rtol=1e-4)

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            resfile = f.name

        try:
            search(
                f=rosenbrock_scaled_func,
                box=[[0, 1], [0, 1]],
                n=8,
                m=8,
                batch=4,
                resfile=resfile,
                executor=get_mock_executor()
            )

            # Verify results file structure
            assert os.path.exists(resfile)

            data = np.loadtxt(resfile, delimiter=',', skiprows=1)
            assert data.shape[0] == 16  # n + m = 8 + 8
            assert data.shape[1] == 3   # 2 params + f_value

            # Results should be sorted
            f_values = data[:, -1]
            assert np.all(f_values[:-1] <= f_values[1:])
        finally:
            if os.path.exists(resfile):
                os.unlink(resfile)

    def test_optimization_with_different_batch_sizes(self):
        """Test that different batch sizes all work correctly."""
        for batch_size in [1, 2, 4]:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                resfile = f.name

            try:
                search(
                    f=sphere_func,
                    box=[[0, 1], [0, 1]],
                    n=4,
                    m=4,
                    batch=batch_size,
                    resfile=resfile,
                    executor=get_mock_executor()
                )

                data = np.loadtxt(resfile, delimiter=',', skiprows=1)
                # Should have points (adjusted to batch size)
                assert len(data) > 0
            finally:
                if os.path.exists(resfile):
                    os.unlink(resfile)
