"""Tests for interleave module"""

import numpy as np
import pytest
from gsm_waveform.interleave import interleave_456_to_4x114


class TestInterleaving:
    """Test block interleaving"""
    
    def test_interleave_output_count(self):
        """Test that interleaving produces 4 bursts"""
        bits = np.zeros(456, dtype=np.uint8)
        bursts = interleave_456_to_4x114(bits)
        
        assert len(bursts) == 4
    
    def test_interleave_output_sizes(self):
        """Test that each burst has 114 bits"""
        bits = np.zeros(456, dtype=np.uint8)
        bursts = interleave_456_to_4x114(bits)
        
        for burst in bursts:
            assert burst.size == 114
    
    def test_interleave_all_zeros(self):
        """Test interleaving all zeros"""
        bits = np.zeros(456, dtype=np.uint8)
        bursts = interleave_456_to_4x114(bits)
        
        for burst in bursts:
            assert np.all(burst == 0)
    
    def test_interleave_pattern(self):
        """Test interleaving pattern correctness"""
        # Create input with known pattern
        bits = np.arange(456, dtype=np.uint8) % 2
        bursts = interleave_456_to_4x114(bits)
        
        # Verify interleaving: column-wise write, row-wise read
        # First 4 bits go to column 0 of rows 0-3
        # bits[0] -> burst[0][0]
        # bits[1] -> burst[1][0]
        # bits[2] -> burst[2][0]
        # bits[3] -> burst[3][0]
        # bits[4] -> burst[0][1]
        # etc.
        
        assert bursts[0][0] == bits[0]
        assert bursts[1][0] == bits[1]
        assert bursts[2][0] == bits[2]
        assert bursts[3][0] == bits[3]
        assert bursts[0][1] == bits[4]
    
    def test_interleave_deterministic(self):
        """Test that interleaving is deterministic"""
        bits = np.random.randint(0, 2, 456, dtype=np.uint8)
        bursts1 = interleave_456_to_4x114(bits)
        bursts2 = interleave_456_to_4x114(bits)
        
        for b1, b2 in zip(bursts1, bursts2):
            assert np.array_equal(b1, b2)
    
    def test_interleave_bit_conservation(self):
        """Test that all bits are conserved during interleaving"""
        bits = np.random.randint(0, 2, 456, dtype=np.uint8)
        bursts = interleave_456_to_4x114(bits)
        
        # Count ones in input
        input_ones = np.sum(bits)
        
        # Count ones in output
        output_ones = sum(np.sum(burst) for burst in bursts)
        
        assert input_ones == output_ones
    
    def test_interleave_wrong_size(self):
        """Test that wrong input size raises assertion"""
        bits = np.zeros(400, dtype=np.uint8)
        with pytest.raises(AssertionError):
            interleave_456_to_4x114(bits)
