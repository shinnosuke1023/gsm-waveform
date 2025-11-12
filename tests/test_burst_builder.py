"""Tests for burst_builder module"""

import numpy as np
import pytest
from gsm_waveform.burst_builder import (
    build_normal_burst,
    build_fcch_burst,
    build_sch_burst,
    build_sch_39bits,
    compute_sch_crc10,
    get_tsc
)
from gsm_waveform.constants import TSC_TABLE


class TestNormalBurst:
    """Test Normal Burst construction"""
    
    def test_normal_burst_size(self):
        """Test that normal burst has 148 bits"""
        data = np.zeros(114, dtype=np.uint8)
        tsc = get_tsc(0)
        burst = build_normal_burst(data, tsc)
        
        assert burst.size == 148
    
    def test_normal_burst_tail_bits(self):
        """Test that tail bits are zero"""
        data = np.ones(114, dtype=np.uint8)
        tsc = get_tsc(0)
        burst = build_normal_burst(data, tsc)
        
        # First 3 bits should be zero (tail)
        assert np.all(burst[:3] == 0)
        # Last 3 bits should be zero (tail)
        assert np.all(burst[-3:] == 0)
    
    def test_normal_burst_tsc_position(self):
        """Test that TSC is in correct position"""
        data = np.zeros(114, dtype=np.uint8)
        tsc = get_tsc(0)
        burst = build_normal_burst(data, tsc)
        
        # TSC should be at position 61-86 (3 tail + 57 data + 1 stealing = 61)
        tsc_start = 3 + 57 + 1
        tsc_end = tsc_start + 26
        assert np.array_equal(burst[tsc_start:tsc_end], tsc)
    
    def test_normal_burst_different_tsc(self):
        """Test building bursts with different TSCs"""
        data = np.zeros(114, dtype=np.uint8)
        
        for i in range(8):
            tsc = get_tsc(i)
            burst = build_normal_burst(data, tsc)
            
            tsc_start = 3 + 57 + 1
            tsc_end = tsc_start + 26
            assert np.array_equal(burst[tsc_start:tsc_end], tsc)
    
    def test_normal_burst_wrong_data_size(self):
        """Test that wrong data size raises assertion"""
        data = np.zeros(100, dtype=np.uint8)
        tsc = get_tsc(0)
        with pytest.raises(AssertionError):
            build_normal_burst(data, tsc)


class TestFCCHBurst:
    """Test FCCH burst construction"""
    
    def test_fcch_burst_size(self):
        """Test that FCCH burst has 148 bits"""
        burst = build_fcch_burst()
        assert burst.size == 148
    
    def test_fcch_burst_all_zeros(self):
        """Test that FCCH burst is all zeros"""
        burst = build_fcch_burst()
        assert np.all(burst == 0)


class TestSCHBurst:
    """Test SCH burst construction"""
    
    def test_sch_crc10_all_zeros(self):
        """Test CRC computation with all zeros"""
        info = np.zeros(25, dtype=np.uint8)
        crc = compute_sch_crc10(info)
        
        assert crc.size == 10
        assert np.all(crc == 0)
    
    def test_sch_crc10_deterministic(self):
        """Test that CRC is deterministic"""
        info = np.random.randint(0, 2, 25, dtype=np.uint8)
        crc1 = compute_sch_crc10(info)
        crc2 = compute_sch_crc10(info)
        
        assert np.array_equal(crc1, crc2)
    
    def test_sch_39bits_size(self):
        """Test that SCH info block has 39 bits"""
        sch_info = build_sch_39bits(bsic=0, fn=0)
        assert sch_info.size == 39
    
    def test_sch_39bits_tail_zeros(self):
        """Test that last 4 bits of SCH info are zeros (tail)"""
        sch_info = build_sch_39bits(bsic=10, fn=100)
        assert np.all(sch_info[-4:] == 0)
    
    def test_sch_burst_size(self):
        """Test that SCH burst has 148 bits"""
        burst = build_sch_burst(bsic=0, fn=0)
        assert burst.size == 148
    
    def test_sch_burst_tail_bits(self):
        """Test that SCH burst has zero tail bits"""
        burst = build_sch_burst(bsic=10, fn=100)
        
        # First 3 bits should be zero (tail)
        assert np.all(burst[:3] == 0)
        # Last 3 bits should be zero (tail)
        assert np.all(burst[-3:] == 0)
    
    def test_sch_burst_different_params(self):
        """Test SCH burst with different parameters"""
        burst1 = build_sch_burst(bsic=0, fn=0)
        burst2 = build_sch_burst(bsic=63, fn=1000)
        
        # Different parameters should produce different bursts
        assert not np.array_equal(burst1, burst2)
    
    def test_sch_burst_deterministic(self):
        """Test that SCH burst is deterministic"""
        burst1 = build_sch_burst(bsic=10, fn=100)
        burst2 = build_sch_burst(bsic=10, fn=100)
        
        assert np.array_equal(burst1, burst2)


class TestTSC:
    """Test Training Sequence Code functions"""
    
    def test_get_tsc_size(self):
        """Test that TSC has 26 bits"""
        for i in range(8):
            tsc = get_tsc(i)
            assert tsc.size == 26
    
    def test_get_tsc_matches_table(self):
        """Test that get_tsc returns correct TSC from table"""
        for i in range(8):
            tsc = get_tsc(i)
            assert np.array_equal(tsc, TSC_TABLE[i])
    
    def test_get_tsc_invalid_index(self):
        """Test that invalid TSC index raises assertion"""
        with pytest.raises(AssertionError):
            get_tsc(8)
        with pytest.raises(AssertionError):
            get_tsc(-1)
    
    def test_tsc_different(self):
        """Test that different TSCs are actually different"""
        tscs = [get_tsc(i) for i in range(8)]
        
        # Check that each TSC is unique
        for i in range(8):
            for j in range(i + 1, 8):
                assert not np.array_equal(tscs[i], tscs[j])
