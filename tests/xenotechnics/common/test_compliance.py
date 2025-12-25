"""
Tests for AbstractSystemCompliance class.

Tests for xenotechnics/common/compliance.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xenotechnics.common import AbstractSystemCompliance, String
from xenotechnics.common.system import AbstractSystem


class MockSystemCompliance(AbstractSystemCompliance):
    """Concrete implementation for testing."""

    def __init__(self, system, string=None, values=None):
        super().__init__(system, string)
        self.values = values or []


class TestAbstractSystemCompliance:
    """Test AbstractSystemCompliance class."""

    @pytest.fixture
    def mock_system(self):
        """Create mock system."""
        system = MagicMock(spec=AbstractSystem)
        system.__len__ = MagicMock(return_value=3)
        return system

    def test_init_with_string(self, mock_system):
        """Test initialization with string."""
        string = String(tokens=("test",))
        compliance = MockSystemCompliance(mock_system, string)

        assert compliance.system == mock_system
        assert compliance.string == string

    def test_init_without_string(self, mock_system):
        """Test initialization without string (for core compliance)."""
        compliance = MockSystemCompliance(mock_system)

        assert compliance.system == mock_system
        assert compliance.string is None

    def test_system_property(self, mock_system):
        """Test system property returns the system."""
        compliance = MockSystemCompliance(mock_system)
        assert compliance.system == mock_system

    def test_len_delegates_to_system(self, mock_system):
        """Test __len__ returns system length."""
        mock_system.__len__.return_value = 5
        compliance = MockSystemCompliance(mock_system)

        assert len(compliance) == 5
        mock_system.__len__.assert_called()

    def test_len_with_different_system_sizes(self, mock_system):
        """Test __len__ with various system sizes."""
        for size in [1, 3, 10, 100]:
            mock_system.__len__.return_value = size
            compliance = MockSystemCompliance(mock_system)
            assert len(compliance) == size


class TestComplianceWithValues:
    """Test compliance with stored values."""

    @pytest.fixture
    def mock_system(self):
        """Create mock system."""
        system = MagicMock(spec=AbstractSystem)
        system.__len__ = MagicMock(return_value=3)
        return system

    def test_store_values(self, mock_system):
        """Test storing compliance values."""
        values = [0.1, 0.5, 0.9]
        compliance = MockSystemCompliance(mock_system, values=values)

        assert compliance.values == values

    def test_compliance_for_specific_string(self, mock_system):
        """Test compliance associated with specific string."""
        string = String(tokens=("The", " ", "cat"))
        values = [0.8, 0.6, 0.4]
        compliance = MockSystemCompliance(mock_system, string, values)

        assert compliance.string == string
        assert compliance.values == values

    def test_core_compliance_has_no_string(self, mock_system):
        """Test core compliance (average) has no string."""
        values = [0.5, 0.5, 0.5]  # Core values
        compliance = MockSystemCompliance(mock_system, string=None, values=values)

        assert compliance.string is None
        assert compliance.values == values
