"""Tests for System Information encoding and decoding"""

import pytest
import numpy as np
from gsm_waveform.system_info import (
    encode_system_information_type3,
    decode_system_information_type3,
    format_system_information,
    _int_to_bits,
    _bits_to_int
)


class TestBitConversion:
    """Test bit conversion helper functions"""
    
    def test_int_to_bits_simple(self):
        """Test converting simple integers to bits"""
        bits = _int_to_bits(5, 8)
        assert len(bits) == 8
        assert list(bits) == [0, 0, 0, 0, 0, 1, 0, 1]
    
    def test_int_to_bits_all_ones(self):
        """Test converting all ones"""
        bits = _int_to_bits(15, 4)
        assert list(bits) == [1, 1, 1, 1]
    
    def test_bits_to_int_simple(self):
        """Test converting bits to integer"""
        bits = np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=np.uint8)
        value = _bits_to_int(bits)
        assert value == 5
    
    def test_bits_to_int_all_ones(self):
        """Test converting all ones"""
        bits = np.array([1, 1, 1, 1], dtype=np.uint8)
        value = _bits_to_int(bits)
        assert value == 15
    
    def test_roundtrip(self):
        """Test roundtrip conversion"""
        original = 12345
        bits = _int_to_bits(original, 16)
        recovered = _bits_to_int(bits)
        assert recovered == original


class TestSystemInformationType3Encoding:
    """Test System Information Type 3 encoding"""
    
    def test_encode_output_size(self):
        """Test that encoding produces 184 bits"""
        info_bits = encode_system_information_type3()
        assert len(info_bits) == 184
        assert info_bits.dtype == np.uint8
    
    def test_encode_with_cell_identity(self):
        """Test encoding with cell identity"""
        cell_id = 12345
        info_bits = encode_system_information_type3(cell_identity=cell_id)
        
        # Decode and verify
        decoded = decode_system_information_type3(info_bits)
        assert decoded['cell_identity'] == cell_id
    
    def test_encode_with_location_area_code(self):
        """Test encoding with location area code"""
        lac = 100
        info_bits = encode_system_information_type3(location_area_code=lac)
        
        decoded = decode_system_information_type3(info_bits)
        assert decoded['location_area_code'] == lac
    
    def test_encode_with_arfcn(self):
        """Test encoding with ARFCN"""
        arfcn = 975
        info_bits = encode_system_information_type3(arfcn=arfcn)
        
        decoded = decode_system_information_type3(info_bits)
        assert decoded['arfcn'] == arfcn
    
    def test_encode_with_neighbor_cells(self):
        """Test encoding with neighbor cells"""
        neighbors = [980, 985, 990, 1000]
        info_bits = encode_system_information_type3(neighbor_cells=neighbors)
        
        decoded = decode_system_information_type3(info_bits)
        assert decoded['neighbor_cells'] == neighbors
    
    def test_encode_with_all_parameters(self):
        """Test encoding with all parameters"""
        info_bits = encode_system_information_type3(
            cell_identity=12345,
            location_area_code=100,
            arfcn=975,
            neighbor_cells=[980, 985, 990]
        )
        
        decoded = decode_system_information_type3(info_bits)
        assert decoded['cell_identity'] == 12345
        assert decoded['location_area_code'] == 100
        assert decoded['arfcn'] == 975
        assert decoded['neighbor_cells'] == [980, 985, 990]
    
    def test_encode_max_neighbor_cells(self):
        """Test encoding with maximum neighbor cells"""
        # Create more than 16 neighbors, should be truncated
        neighbors = list(range(1, 20))
        info_bits = encode_system_information_type3(neighbor_cells=neighbors)
        
        decoded = decode_system_information_type3(info_bits)
        # Should only have first 16
        assert len(decoded['neighbor_cells']) <= 16


class TestSystemInformationType3Decoding:
    """Test System Information Type 3 decoding"""
    
    def test_decode_all_zeros(self):
        """Test decoding all zeros"""
        info_bits = np.zeros(184, dtype=np.uint8)
        decoded = decode_system_information_type3(info_bits)
        
        assert decoded['valid'] is True
        assert decoded['cell_identity'] == 0
        assert decoded['location_area_code'] == 0
        assert decoded['arfcn'] == 0
        assert decoded['neighbor_cells'] == []
    
    def test_decode_wrong_size(self):
        """Test decoding with wrong size"""
        info_bits = np.zeros(100, dtype=np.uint8)
        decoded = decode_system_information_type3(info_bits)
        
        assert decoded['valid'] is False
        assert 'error' in decoded
    
    def test_decode_roundtrip(self):
        """Test encoding then decoding"""
        # Create message
        info_bits = encode_system_information_type3(
            cell_identity=54321,
            location_area_code=200,
            arfcn=123,
            neighbor_cells=[100, 200, 300]
        )
        
        # Decode
        decoded = decode_system_information_type3(info_bits)
        
        # Verify
        assert decoded['valid'] is True
        assert decoded['cell_identity'] == 54321
        assert decoded['location_area_code'] == 200
        assert decoded['arfcn'] == 123
        assert decoded['neighbor_cells'] == [100, 200, 300]


class TestSystemInformationFormatting:
    """Test System Information formatting"""
    
    def test_format_valid_message(self):
        """Test formatting valid System Information"""
        info_bits = encode_system_information_type3(
            cell_identity=12345,
            location_area_code=100,
            arfcn=975,
            neighbor_cells=[980, 985]
        )
        
        formatted = format_system_information(info_bits)
        
        # Check that key information is present
        assert '12345' in formatted
        assert '100' in formatted
        assert '975' in formatted
        assert '980' in formatted
        assert '985' in formatted
    
    def test_format_invalid_message(self):
        """Test formatting invalid message"""
        info_bits = np.zeros(100, dtype=np.uint8)
        formatted = format_system_information(info_bits)
        
        assert 'Invalid' in formatted or 'error' in formatted.lower()


class TestEndToEndPipeline:
    """Test complete encoding/decoding pipeline with BCCH"""
    
    def test_system_info_through_bcch_pipeline(self):
        """Test System Information through complete BCCH encode/decode"""
        from gsm_waveform import (
            make_bcch_encoded_456,
            interleave_456_to_4x114,
            decode_bcch_pipeline
        )
        
        # Encode System Information
        # Note: cell_identity and location_area_code are 16-bit fields (0-65535)
        info_bits = encode_system_information_type3(
            cell_identity=54321,
            location_area_code=555,
            arfcn=888,
            neighbor_cells=[900, 910, 920, 930]
        )
        
        # Run through BCCH encoding
        encoded_456 = make_bcch_encoded_456(info_bits)
        data_bursts = interleave_456_to_4x114(encoded_456)
        
        # Decode
        decoded_info, valid = decode_bcch_pipeline(data_bursts)
        
        # Verify decoding succeeded
        assert valid is True
        assert np.array_equal(decoded_info, info_bits)
        
        # Verify System Information content
        decoded_si = decode_system_information_type3(decoded_info)
        assert decoded_si['cell_identity'] == 54321
        assert decoded_si['location_area_code'] == 555
        assert decoded_si['arfcn'] == 888
        assert decoded_si['neighbor_cells'] == [900, 910, 920, 930]
