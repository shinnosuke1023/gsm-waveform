"""Tests for modulator module"""

import numpy as np
import pytest
from gsm_waveform.modulator import (
    design_gaussian_filter,
    gmsk_modulate,
    modulate_burst_sequence
)
from gsm_waveform.constants import BT, OSR_DEFAULT, GAUSSIAN_FILTER_SPAN


class TestGaussianFilter:
    """Test Gaussian filter design"""
    
    def test_gaussian_filter_length(self):
        """Test that filter has correct length"""
        h = design_gaussian_filter(BT, OSR_DEFAULT, GAUSSIAN_FILTER_SPAN)
        expected_len = GAUSSIAN_FILTER_SPAN * OSR_DEFAULT + 1
        assert len(h) == expected_len
    
    def test_gaussian_filter_normalized(self):
        """Test that filter is normalized"""
        h = design_gaussian_filter(BT, OSR_DEFAULT, GAUSSIAN_FILTER_SPAN)
        assert np.isclose(np.sum(h), 1.0)
    
    def test_gaussian_filter_symmetric(self):
        """Test that filter is symmetric"""
        h = design_gaussian_filter(BT, OSR_DEFAULT, GAUSSIAN_FILTER_SPAN)
        assert np.allclose(h, h[::-1])
    
    def test_gaussian_filter_peak_center(self):
        """Test that filter peak is at center"""
        h = design_gaussian_filter(BT, OSR_DEFAULT, GAUSSIAN_FILTER_SPAN)
        center_idx = len(h) // 2
        assert h[center_idx] == np.max(h)


class TestGMSKModulation:
    """Test GMSK modulation"""
    
    def test_gmsk_output_type(self):
        """Test that GMSK output is complex64"""
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        iq = gmsk_modulate(bits)
        assert iq.dtype == np.complex64
    
    def test_gmsk_output_length(self):
        """Test that output length is correct"""
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        iq = gmsk_modulate(bits, osr=8)
        # Output should be approximately bits * osr
        # (may vary slightly due to filtering)
        assert len(iq) == len(bits) * 8
    
    def test_gmsk_unit_magnitude(self):
        """Test that IQ samples have approximately unit magnitude"""
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        iq = gmsk_modulate(bits)
        magnitudes = np.abs(iq)
        
        # All magnitudes should be close to 1.0
        assert np.all(magnitudes > 0.9)
        assert np.all(magnitudes < 1.1)
    
    def test_gmsk_all_zeros(self):
        """Test GMSK modulation with all zeros"""
        bits = np.zeros(10, dtype=np.uint8)
        iq = gmsk_modulate(bits)
        
        # Should produce valid complex samples
        assert len(iq) > 0
        assert not np.any(np.isnan(iq))
        assert not np.any(np.isinf(iq))
    
    def test_gmsk_all_ones(self):
        """Test GMSK modulation with all ones"""
        bits = np.ones(10, dtype=np.uint8)
        iq = gmsk_modulate(bits)
        
        # Should produce valid complex samples
        assert len(iq) > 0
        assert not np.any(np.isnan(iq))
        assert not np.any(np.isinf(iq))
    
    def test_gmsk_deterministic(self):
        """Test that modulation is deterministic"""
        bits = np.random.randint(0, 2, 20, dtype=np.uint8)
        iq1 = gmsk_modulate(bits)
        iq2 = gmsk_modulate(bits)
        
        assert np.array_equal(iq1, iq2)
    
    def test_gmsk_different_inputs(self):
        """Test that different inputs produce different outputs"""
        bits1 = np.zeros(10, dtype=np.uint8)
        bits2 = np.ones(10, dtype=np.uint8)
        
        iq1 = gmsk_modulate(bits1)
        iq2 = gmsk_modulate(bits2)
        
        # Different inputs should produce different outputs
        assert not np.array_equal(iq1, iq2)


class TestBurstSequenceModulation:
    """Test burst sequence modulation"""
    
    def test_burst_sequence_single(self):
        """Test modulating a single burst"""
        burst = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        iq = modulate_burst_sequence([burst])
        
        assert len(iq) > 0
        assert iq.dtype == np.complex64
    
    def test_burst_sequence_multiple(self):
        """Test modulating multiple bursts"""
        burst1 = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        burst2 = np.array([1, 0, 0, 1, 0], dtype=np.uint8)
        
        iq = modulate_burst_sequence([burst1, burst2])
        
        assert len(iq) > 0
        assert iq.dtype == np.complex64
    
    def test_burst_sequence_with_guard(self):
        """Test modulating bursts with guard samples"""
        burst1 = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        burst2 = np.array([1, 0, 0, 1, 0], dtype=np.uint8)
        
        iq_no_guard = modulate_burst_sequence([burst1, burst2], guard_samples=0)
        iq_with_guard = modulate_burst_sequence([burst1, burst2], guard_samples=10)
        
        # With guard samples, output should be longer
        assert len(iq_with_guard) > len(iq_no_guard)
    
    def test_burst_sequence_deterministic(self):
        """Test that burst sequence modulation is deterministic"""
        bursts = [
            np.random.randint(0, 2, 20, dtype=np.uint8),
            np.random.randint(0, 2, 20, dtype=np.uint8)
        ]
        
        iq1 = modulate_burst_sequence(bursts)
        iq2 = modulate_burst_sequence(bursts)
        
        assert np.array_equal(iq1, iq2)
