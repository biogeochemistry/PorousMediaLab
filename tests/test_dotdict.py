"""Tests for DotDict utility class."""

import pytest
from porousmedialab.dotdict import DotDict


class TestDotDict:
    """Tests for DotDict dictionary with dot notation access."""

    def test_dot_notation_get(self):
        """Verify that attributes can be accessed via dot notation."""
        d = DotDict({'key': 'value', 'number': 42})
        assert d.key == 'value'
        assert d.number == 42

    def test_dot_notation_set(self):
        """Verify that attributes can be set via dot notation."""
        d = DotDict()
        d.key = 'value'
        d.number = 42
        assert d['key'] == 'value'
        assert d['number'] == 42

    def test_dot_notation_delete(self):
        """Verify that attributes can be deleted via dot notation."""
        d = DotDict({'key': 'value'})
        del d.key
        assert 'key' not in d

    def test_missing_key_returns_none(self):
        """Verify that accessing a missing key returns None (not KeyError)."""
        d = DotDict()
        assert d.nonexistent is None

    def test_nested_dotdict(self):
        """Verify that nested DotDicts work correctly."""
        d = DotDict()
        d.nested = DotDict({'inner': 'value'})
        assert d.nested.inner == 'value'

    def test_dict_methods_preserved(self):
        """Verify that standard dict methods still work."""
        d = DotDict({'a': 1, 'b': 2})
        assert set(d.keys()) == {'a', 'b'}
        assert set(d.values()) == {1, 2}
        assert set(d.items()) == {('a', 1), ('b', 2)}
        assert len(d) == 2

    def test_iteration(self):
        """Verify that iteration over DotDict works."""
        d = DotDict({'a': 1, 'b': 2, 'c': 3})
        keys = set(k for k in d)
        assert keys == {'a', 'b', 'c'}

    def test_update_method(self):
        """Verify that update method works correctly."""
        d = DotDict({'a': 1})
        d.update({'b': 2, 'c': 3})
        assert d.a == 1
        assert d.b == 2
        assert d.c == 3
