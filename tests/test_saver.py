"""Tests for HDF5 save/load functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from porousmedialab.saver import (
    save_dict_to_hdf5,
    load_dict_from_hdf5,
    recursively_save_dict_contents_to_group,
    recursively_load_dict_contents_from_group
)


class TestSaveDictToHdf5:
    """Tests for saving dictionaries to HDF5."""

    def test_save_simple_dict(self, temp_hdf5_file):
        """Should save a simple dictionary."""
        data = {'x': np.array([1, 2, 3]), 'y': np.array([4.0, 5.0, 6.0])}
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        assert temp_hdf5_file.exists()

    def test_save_string(self, temp_hdf5_file):
        """Should save string values."""
        data = {'name': 'test_string'}
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        # HDF5 stores strings as bytes
        assert loaded['name'] == b'test_string' or loaded['name'] == 'test_string'

    def test_save_bytes(self, temp_hdf5_file):
        """Should save byte strings."""
        data = {'data': b'byte_string'}
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        assert loaded['data'] == b'byte_string'

    def test_save_numpy_array(self, temp_hdf5_file):
        """Should save numpy arrays."""
        data = {'arr': np.arange(100).reshape(10, 10)}
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        assert_array_equal(loaded['arr'], data['arr'])

    def test_save_nested_dict(self, temp_hdf5_file):
        """Should save nested dictionaries."""
        data = {
            'level1': {
                'level2': {
                    'data': np.array([1, 2, 3])
                }
            }
        }
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        assert_array_equal(loaded['level1']['level2']['data'], [1, 2, 3])

    def test_save_int64(self, temp_hdf5_file):
        """Should save np.int64 values."""
        data = {'value': np.int64(42)}
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        assert loaded['value'] == 42

    def test_save_float64(self, temp_hdf5_file):
        """Should save np.float64 values."""
        data = {'value': np.float64(3.14159)}
        save_dict_to_hdf5(data, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        assert_allclose(loaded['value'], 3.14159)

    def test_save_unsupported_type_raises(self, temp_hdf5_file):
        """Should raise ValueError for unsupported types."""
        data = {'bad': [1, 2, 3]}  # Python list not supported
        with pytest.raises(ValueError, match="Cannot save"):
            save_dict_to_hdf5(data, str(temp_hdf5_file))


class TestLoadDictFromHdf5:
    """Tests for loading dictionaries from HDF5."""

    def test_roundtrip_simple(self, temp_hdf5_file):
        """Data should survive save/load roundtrip."""
        original = {
            'a': np.array([1.0, 2.0, 3.0]),
            'b': np.array([[1, 2], [3, 4]])
        }
        save_dict_to_hdf5(original, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))

        assert_array_equal(loaded['a'], original['a'])
        assert_array_equal(loaded['b'], original['b'])

    def test_roundtrip_nested(self, temp_hdf5_file):
        """Nested dicts should survive roundtrip."""
        original = {
            'outer': {
                'inner': np.arange(10)
            },
            'flat': np.array([1.0])
        }
        save_dict_to_hdf5(original, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))

        assert_array_equal(loaded['outer']['inner'], original['outer']['inner'])
        assert_array_equal(loaded['flat'], original['flat'])

    def test_roundtrip_mixed_types(self, temp_hdf5_file):
        """Mixed types should survive roundtrip."""
        original = {
            'string': 'hello',
            'array': np.array([1, 2, 3]),
            'scalar': np.float64(42.0)
        }
        save_dict_to_hdf5(original, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))

        assert_array_equal(loaded['array'], original['array'])
        assert_allclose(loaded['scalar'], 42.0)


class TestHdf5Integration:
    """Integration tests simulating real usage patterns."""

    def test_save_simulation_results(self, temp_hdf5_file):
        """Test saving typical simulation results."""
        results = {
            'time': np.linspace(0, 1, 101),
            'concentrations': {
                'O2': np.random.rand(10, 101),
                'CO2': np.random.rand(10, 101)
            },
            'parameters': {
                'k': '1.5',
                'D': '1e-5'
            }
        }

        save_dict_to_hdf5(results, str(temp_hdf5_file))
        loaded = load_dict_from_hdf5(str(temp_hdf5_file))

        assert_array_equal(loaded['time'], results['time'])
        assert_array_equal(
            loaded['concentrations']['O2'],
            results['concentrations']['O2']
        )

    def test_overwrite_existing_file(self, temp_hdf5_file):
        """Saving to existing file should overwrite."""
        data1 = {'x': np.array([1, 2, 3])}
        save_dict_to_hdf5(data1, str(temp_hdf5_file))

        data2 = {'y': np.array([4, 5, 6])}
        save_dict_to_hdf5(data2, str(temp_hdf5_file))

        loaded = load_dict_from_hdf5(str(temp_hdf5_file))
        assert 'y' in loaded
        # x should be gone since file was overwritten
        assert 'x' not in loaded
