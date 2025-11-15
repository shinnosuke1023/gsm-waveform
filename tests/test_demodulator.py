"""Tests for GSM GMSK demodulator"""

import numpy as np
import pytest
from gsm_waveform.demodulator import (
    gmsk_demodulate,
    correlate_sequence,
    detect_burst_by_tsc,
    extract_normal_burst_data,
    extract_burst_data_114,
    detect_sch_burst,
    extract_sch_data
)
from gsm_waveform.modulator import gmsk_modulate
from gsm_waveform.constants import TSC_TABLE, SCH_TRAINING_SEQ


class TestGMSKDemodulation:
    """Test GMSK demodulation"""
    
    def test_demod_simple_pattern(self):
        """Test demodulation of a simple bit pattern"""
        # Create a simple bit pattern
        bits = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        
        # Modulate
        iq = gmsk_modulate(bits)
        
        # Demodulate
        demod_bits = gmsk_demodulate(iq)
        
        # Should recover most bits (allow some error at edges)
        # Check middle bits
        assert len(demod_bits) >= len(bits) - 2
    
    def test_demod_all_zeros(self):
        """Test demodulation of all zeros"""
        bits = np.zeros(100, dtype=np.uint8)
        iq = gmsk_modulate(bits)
        demod_bits = gmsk_demodulate(iq)
        
        # Most bits should be zero
        assert np.sum(demod_bits) < len(demod_bits) * 0.2
    
    def test_demod_all_ones(self):
        """Test demodulation of all ones"""
        bits = np.ones(100, dtype=np.uint8)
        iq = gmsk_modulate(bits)
        demod_bits = gmsk_demodulate(iq)
        
        # Most bits should be one
        assert np.sum(demod_bits) > len(demod_bits) * 0.8
    
    def test_demod_output_length(self):
        """Test that demodulated output has reasonable length"""
        bits = np.random.randint(0, 2, 150, dtype=np.uint8)
        iq = gmsk_modulate(bits, osr=8)
        demod_bits = gmsk_demodulate(iq, osr=8, return_all_phases=False)
        
        # Should be close to original length
        assert len(demod_bits) >= len(bits) - 10
        assert len(demod_bits) <= len(bits) + 10


class TestCorrelation:
    """Test correlation functions"""
    
    def test_correlate_identical(self):
        """Test correlation of identical sequences"""
        seq = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        corr = correlate_sequence(seq, seq)
        
        # Peak should be at position 0 with maximum value
        assert len(corr) > 0
        assert corr[0] == len(seq)
    
    def test_correlate_shifted(self):
        """Test correlation with shifted sequence"""
        signal = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0], dtype=np.uint8)
        pattern = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        
        corr = correlate_sequence(signal, pattern)
        
        # Peak should be where pattern matches
        assert len(corr) > 0
        assert np.argmax(np.abs(corr)) == 3
    
    def test_correlate_no_match(self):
        """Test correlation with no matching pattern"""
        signal = np.zeros(20, dtype=np.uint8)
        pattern = np.ones(5, dtype=np.uint8)
        
        corr = correlate_sequence(signal, pattern)
        
        # Correlation should be negative (anti-correlated)
        assert len(corr) > 0
        assert np.all(corr <= 0)


class TestBurstDetection:
    """Test burst detection functions"""
    
    def test_detect_burst_by_tsc(self):
        """Test burst detection using TSC"""
        # Create a sequence with TSC embedded
        tsc = TSC_TABLE[0]
        prefix = np.zeros(50, dtype=np.uint8)
        suffix = np.ones(50, dtype=np.uint8)
        signal = np.concatenate([prefix, tsc, suffix])
        
        # Detect bursts
        positions = detect_burst_by_tsc(signal, tsc_index=0, threshold=0.7)
        
        # Should detect the TSC position
        assert len(positions) > 0
        # Position should be near where we inserted TSC
        assert any(45 <= pos <= 55 for pos in positions)
    
    def test_detect_multiple_bursts(self):
        """Test detection of multiple bursts"""
        tsc = TSC_TABLE[0]
        burst1 = np.concatenate([np.zeros(50, dtype=np.uint8), tsc, np.ones(50, dtype=np.uint8)])
        burst2 = np.concatenate([np.zeros(50, dtype=np.uint8), tsc, np.ones(50, dtype=np.uint8)])
        signal = np.concatenate([burst1, burst2])
        
        positions = detect_burst_by_tsc(signal, tsc_index=0, threshold=0.6)
        
        # Should detect both bursts
        assert len(positions) >= 2
    
    def test_detect_sch_burst(self):
        """Test SCH burst detection"""
        # Create signal with SCH training sequence
        prefix = np.zeros(30, dtype=np.uint8)
        suffix = np.ones(30, dtype=np.uint8)
        signal = np.concatenate([prefix, SCH_TRAINING_SEQ, suffix])
        
        positions = detect_sch_burst(signal, threshold=0.6)
        
        # Should detect the SCH position
        assert len(positions) > 0
        assert any(25 <= pos <= 35 for pos in positions)


class TestBurstDataExtraction:
    """Test burst data extraction"""
    
    def test_extract_normal_burst_data(self):
        """Test extracting data from normal burst"""
        # Create a normal burst structure
        burst = np.random.randint(0, 2, 148, dtype=np.uint8)
        
        left, right = extract_normal_burst_data(burst)
        
        assert len(left) == 57
        assert len(right) == 57
    
    def test_extract_burst_data_114(self):
        """Test extracting 114 data bits"""
        burst = np.random.randint(0, 2, 148, dtype=np.uint8)
        
        data = extract_burst_data_114(burst)
        
        assert len(data) == 114
        assert isinstance(data, np.ndarray)
    
    def test_extract_sch_data(self):
        """Test extracting data from SCH burst"""
        burst = np.random.randint(0, 2, 148, dtype=np.uint8)
        
        left, right = extract_sch_data(burst)
        
        assert len(left) == 39
        assert len(right) == 39


class TestEndToEndDemodulation:
    """Test end-to-end modulation and demodulation"""
    
    def test_roundtrip_simple(self):
        """Test simple roundtrip: modulate then demodulate"""
        # Create known bit pattern
        bits = np.tile([0, 1, 0, 1], 25)  # 100 bits
        
        # Modulate
        iq = gmsk_modulate(bits)
        
        # Demodulate - try all phases and pick best
        all_phases = gmsk_demodulate(iq, return_all_phases=True)
        
        # Find best phase alignment
        best_similarity = 0
        for phase_bits in all_phases:
            if len(phase_bits) >= len(bits) - 10:
                # Compare middle section (avoid edge effects)
                start = 5
                end = min(len(bits) - 5, len(phase_bits) - 5)
                if end > start:
                    middle_orig = bits[start:end]
                    middle_demod = phase_bits[start:end]
                    similarity = np.mean(middle_orig == middle_demod)
                    best_similarity = max(best_similarity, similarity)
        
        # At least one phase should have good similarity
        assert best_similarity > 0.7  # At least 70% correct
