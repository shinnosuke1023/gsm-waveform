"""Tests for encoding module"""

import numpy as np
import pytest
from gsm_waveform.encoding import (
    fire_encode_184,
    append_block_tail,
    convolutional_encode,
    make_bcch_encoded_456
)


class TestFireEncoding:
    """Test FIRE code encoding"""
    
    def test_fire_encode_all_zeros(self):
        """Test FIRE encoding with all-zero input"""
        info = np.zeros(184, dtype=np.uint8)
        encoded = fire_encode_184(info)
        
        assert encoded.size == 224
        assert np.all(encoded[:184] == 0)
        # Parity should also be zero for all-zero input
        assert np.all(encoded[184:] == 0)
    
    def test_fire_encode_all_ones(self):
        """Test FIRE encoding with all-one input"""
        info = np.ones(184, dtype=np.uint8)
        encoded = fire_encode_184(info)
        
        assert encoded.size == 224
        assert np.all(encoded[:184] == 1)
        # Parity should be non-trivial
    
    def test_fire_encode_deterministic(self):
        """Test that FIRE encoding is deterministic"""
        info = np.random.randint(0, 2, 184, dtype=np.uint8)
        encoded1 = fire_encode_184(info)
        encoded2 = fire_encode_184(info)
        
        assert np.array_equal(encoded1, encoded2)
    
    def test_fire_encode_wrong_size(self):
        """Test that wrong input size raises assertion"""
        info = np.zeros(100, dtype=np.uint8)
        with pytest.raises(AssertionError):
            fire_encode_184(info)


class TestBlockTail:
    """Test tail bit appending"""
    
    def test_append_tail_default(self):
        """Test appending default 4 tail bits"""
        bits = np.ones(224, dtype=np.uint8)
        with_tail = append_block_tail(bits)
        
        assert with_tail.size == 228
        assert np.all(with_tail[:224] == 1)
        assert np.all(with_tail[224:] == 0)
    
    def test_append_tail_custom(self):
        """Test appending custom number of tail bits"""
        bits = np.ones(100, dtype=np.uint8)
        with_tail = append_block_tail(bits, tail_len=8)
        
        assert with_tail.size == 108
        assert np.all(with_tail[:100] == 1)
        assert np.all(with_tail[100:] == 0)


class TestConvolutionalEncoder:
    """Test convolutional encoder"""
    
    def test_conv_all_zeros(self):
        """Test convolutional encoding with all zeros"""
        bits = np.zeros(10, dtype=np.uint8)
        encoded = convolutional_encode(bits)
        
        assert encoded.size == 20
        # All zeros input should produce all zeros output
        assert np.all(encoded == 0)
    
    def test_conv_output_size(self):
        """Test that output is twice the input size"""
        for size in [10, 50, 100, 228]:
            bits = np.random.randint(0, 2, size, dtype=np.uint8)
            encoded = convolutional_encode(bits)
            assert encoded.size == size * 2
    
    def test_conv_deterministic(self):
        """Test that encoding is deterministic"""
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        encoded1 = convolutional_encode(bits)
        encoded2 = convolutional_encode(bits)
        
        assert np.array_equal(encoded1, encoded2)
    
    def test_conv_single_bit(self):
        """Test encoding of single bit"""
        # Input: single 1 bit
        bits = np.array([1], dtype=np.uint8)
        encoded = convolutional_encode(bits)
        
        assert encoded.size == 2
        # For generators (0o23, 0o35) = (10011, 11101)
        # Input 1 with zero state should give specific output
        # G0 = 0o23 = 10011 -> taps at D^0,D^1,D^4 -> with input=1, state=[0,0,0,0]: 1^0^0^0^0 = 1
        # G1 = 0o35 = 11101 -> taps at D^0,D^2,D^3,D^4 -> with input=1, state=[0,0,0,0]: 1^0^0^0^0 = 1
        assert encoded[0] == 1  # G0 output
        assert encoded[1] == 1  # G1 output


class TestBCCHPipeline:
    """Test complete BCCH encoding pipeline"""
    
    def test_bcch_pipeline_output_size(self):
        """Test that BCCH pipeline produces 456 bits"""
        info = np.zeros(184, dtype=np.uint8)
        encoded = make_bcch_encoded_456(info)
        
        assert encoded.size == 456
    
    def test_bcch_pipeline_all_zeros(self):
        """Test BCCH pipeline with all zeros"""
        info = np.zeros(184, dtype=np.uint8)
        encoded = make_bcch_encoded_456(info)
        
        # All zeros should produce all zeros through the pipeline
        assert np.all(encoded == 0)
    
    def test_bcch_pipeline_deterministic(self):
        """Test that pipeline is deterministic"""
        info = np.random.randint(0, 2, 184, dtype=np.uint8)
        encoded1 = make_bcch_encoded_456(info)
        encoded2 = make_bcch_encoded_456(info)
        
        assert np.array_equal(encoded1, encoded2)
    
    def test_bcch_pipeline_wrong_size(self):
        """Test that wrong input size raises assertion"""
        info = np.zeros(100, dtype=np.uint8)
        with pytest.raises(AssertionError):
            make_bcch_encoded_456(info)
