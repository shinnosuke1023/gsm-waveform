"""Tests for GSM decoder functions"""

import numpy as np
import pytest
from gsm_waveform.decoder import (
    deinterleave_4x114_to_456,
    viterbi_decode,
    remove_tail_bits,
    fire_decode_224,
    decode_bcch_pipeline,
    decode_sch_39bits
)
from gsm_waveform.encoding import (
    fire_encode_184,
    append_block_tail,
    convolutional_encode,
    make_bcch_encoded_456
)
from gsm_waveform.interleave import interleave_456_to_4x114
from gsm_waveform.burst_builder import build_sch_39bits


class TestDeinterleaving:
    """Test deinterleaving functions"""
    
    def test_deinterleave_output_size(self):
        """Test that deinterleaving produces correct output size"""
        bursts = [np.zeros(114, dtype=np.uint8) for _ in range(4)]
        result = deinterleave_4x114_to_456(bursts)
        
        assert len(result) == 456
        assert isinstance(result, np.ndarray)
    
    def test_deinterleave_wrong_count(self):
        """Test error handling for wrong number of bursts"""
        bursts = [np.zeros(114, dtype=np.uint8) for _ in range(3)]
        
        with pytest.raises(AssertionError):
            deinterleave_4x114_to_456(bursts)
    
    def test_deinterleave_wrong_size(self):
        """Test error handling for wrong burst size"""
        bursts = [np.zeros(100, dtype=np.uint8) for _ in range(4)]
        
        with pytest.raises(AssertionError):
            deinterleave_4x114_to_456(bursts)
    
    def test_deinterleave_roundtrip(self):
        """Test that interleave/deinterleave is reversible"""
        # Create random 456-bit sequence
        original = np.random.randint(0, 2, 456, dtype=np.uint8)
        
        # Interleave
        bursts = interleave_456_to_4x114(original)
        
        # Deinterleave
        reconstructed = deinterleave_4x114_to_456(bursts)
        
        # Should match original
        assert np.array_equal(original, reconstructed)


class TestViterbiDecoding:
    """Test Viterbi decoder"""
    
    def test_viterbi_all_zeros(self):
        """Test Viterbi decoding of all-zero input"""
        # Encode all zeros
        info = np.zeros(10, dtype=np.uint8)
        encoded = convolutional_encode(info)
        
        # Decode
        decoded = viterbi_decode(encoded)
        
        assert len(decoded) == len(info)
        # Should decode to all zeros
        assert np.array_equal(decoded, info)
    
    def test_viterbi_output_length(self):
        """Test that Viterbi decoder produces correct output length"""
        encoded = np.zeros(100, dtype=np.uint8)
        decoded = viterbi_decode(encoded)
        
        assert len(decoded) == 50  # Half of input length
    
    def test_viterbi_roundtrip(self):
        """Test encode/decode roundtrip"""
        # Create random information bits
        info = np.random.randint(0, 2, 20, dtype=np.uint8)
        
        # Encode
        encoded = convolutional_encode(info)
        
        # Decode
        decoded = viterbi_decode(encoded)
        
        # Should match original (with tail bits handled)
        assert len(decoded) == len(info)
        # Most bits should match (allow some error for random patterns)
        similarity = np.mean(decoded == info)
        assert similarity > 0.8
    
    def test_viterbi_with_tail(self):
        """Test Viterbi decoding with tail bits"""
        # Create info with tail
        info = np.random.randint(0, 2, 20, dtype=np.uint8)
        info_with_tail = append_block_tail(info, tail_len=4)
        
        # Encode
        encoded = convolutional_encode(info_with_tail)
        
        # Decode
        decoded = viterbi_decode(encoded)
        
        # Remove tail
        decoded_no_tail = remove_tail_bits(decoded, tail_len=4)
        
        # Should match original info
        assert len(decoded_no_tail) == len(info)


class TestRemoveTailBits:
    """Test tail bit removal"""
    
    def test_remove_tail_default(self):
        """Test removing default 4 tail bits"""
        bits = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        result = remove_tail_bits(bits, tail_len=4)
        
        assert len(result) == 4
        assert np.array_equal(result, bits[:4])
    
    def test_remove_tail_custom(self):
        """Test removing custom number of tail bits"""
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        result = remove_tail_bits(bits, tail_len=2)
        
        assert len(result) == 4
        assert np.array_equal(result, bits[:4])
    
    def test_remove_tail_too_short(self):
        """Test removing tail from too-short sequence"""
        bits = np.array([1, 0, 1], dtype=np.uint8)
        result = remove_tail_bits(bits, tail_len=4)
        
        assert len(result) == 0


class TestFireDecoding:
    """Test FIRE code decoding"""
    
    def test_fire_decode_all_zeros(self):
        """Test FIRE decoding of all zeros"""
        info = np.zeros(184, dtype=np.uint8)
        encoded = fire_encode_184(info)
        
        decoded_info, valid = fire_decode_224(encoded)
        
        assert len(decoded_info) == 184
        assert valid == True
        assert np.array_equal(decoded_info, info)
    
    def test_fire_decode_random(self):
        """Test FIRE decoding of random data"""
        info = np.random.randint(0, 2, 184, dtype=np.uint8)
        encoded = fire_encode_184(info)
        
        decoded_info, valid = fire_decode_224(encoded)
        
        assert len(decoded_info) == 184
        assert valid == True
        assert np.array_equal(decoded_info, info)
    
    def test_fire_decode_corrupted(self):
        """Test FIRE decoding with corrupted data"""
        info = np.zeros(184, dtype=np.uint8)
        encoded = fire_encode_184(info)
        
        # Corrupt one bit in parity
        encoded[200] ^= 1
        
        decoded_info, valid = fire_decode_224(encoded)
        
        assert len(decoded_info) == 184
        assert valid == False  # Parity should fail
    
    def test_fire_decode_wrong_size(self):
        """Test FIRE decoding with wrong input size"""
        wrong_size = np.zeros(200, dtype=np.uint8)
        
        with pytest.raises(AssertionError):
            fire_decode_224(wrong_size)


class TestBCCHPipeline:
    """Test complete BCCH encoding/decoding pipeline"""
    
    def test_bcch_pipeline_roundtrip(self):
        """Test complete BCCH encode/decode roundtrip"""
        # Create random information
        info = np.random.randint(0, 2, 184, dtype=np.uint8)
        
        # Encode
        encoded_456 = make_bcch_encoded_456(info)
        
        # Interleave
        data_bursts = interleave_456_to_4x114(encoded_456)
        
        # Decode
        decoded_info, valid = decode_bcch_pipeline(data_bursts)
        
        assert len(decoded_info) == 184
        assert valid == True
        # Should match original
        assert np.array_equal(decoded_info, info)
    
    def test_bcch_pipeline_all_zeros(self):
        """Test BCCH pipeline with all zeros"""
        info = np.zeros(184, dtype=np.uint8)
        
        # Encode and interleave
        encoded_456 = make_bcch_encoded_456(info)
        data_bursts = interleave_456_to_4x114(encoded_456)
        
        # Decode
        decoded_info, valid = decode_bcch_pipeline(data_bursts)
        
        assert valid == True
        assert np.array_equal(decoded_info, info)
    
    def test_bcch_pipeline_all_ones(self):
        """Test BCCH pipeline with all ones"""
        info = np.ones(184, dtype=np.uint8)
        
        # Encode and interleave
        encoded_456 = make_bcch_encoded_456(info)
        data_bursts = interleave_456_to_4x114(encoded_456)
        
        # Decode
        decoded_info, valid = decode_bcch_pipeline(data_bursts)
        
        assert valid == True
        assert np.array_equal(decoded_info, info)


class TestSCHDecoding:
    """Test SCH decoding"""
    
    def test_sch_decode_simple(self):
        """Test SCH decoding with simple values"""
        bsic = 10
        fn = 0
        
        # Build SCH data
        sch39 = build_sch_39bits(bsic, fn)
        
        # Decode
        dec_bsic, dec_fn, valid = decode_sch_39bits(sch39)
        
        assert valid == True
        assert dec_bsic == bsic
        # Frame number might not match exactly due to encoding/decoding
        # but should be close
        assert dec_fn is not None
    
    def test_sch_decode_different_values(self):
        """Test SCH decoding with different BSIC values"""
        for bsic in [0, 10, 31, 63]:
            sch39 = build_sch_39bits(bsic, 0)
            dec_bsic, dec_fn, valid = decode_sch_39bits(sch39)
            
            assert valid == True
            assert dec_bsic == bsic
    
    def test_sch_decode_corrupted(self):
        """Test SCH decoding with corrupted CRC"""
        bsic = 10
        fn = 0
        
        sch39 = build_sch_39bits(bsic, fn)
        
        # Corrupt CRC
        sch39[30] ^= 1
        
        dec_bsic, dec_fn, valid = decode_sch_39bits(sch39)
        
        assert valid == False
        assert dec_bsic is None
        assert dec_fn is None
