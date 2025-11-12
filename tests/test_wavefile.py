"""Tests for wavefile module"""

import numpy as np
import pytest
import tempfile
import os
from gsm_waveform.wavefile import (
    write_complex_iq,
    read_complex_iq,
    write_iq_interleaved_float,
    get_sample_info
)


class TestComplexIQIO:
    """Test complex IQ file I/O"""
    
    def test_write_read_complex64(self):
        """Test writing and reading complex64 IQ samples"""
        # Create test data
        iq_out = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            # Write
            write_complex_iq(filename, iq_out, dtype='complex64')
            
            # Read
            iq_in = read_complex_iq(filename, dtype='complex64')
            
            # Verify
            assert np.array_equal(iq_in, iq_out)
        finally:
            os.unlink(filename)
    
    def test_write_read_complex128(self):
        """Test writing and reading complex128 IQ samples"""
        # Create test data
        iq_out = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex128)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            # Write
            write_complex_iq(filename, iq_out, dtype='complex128')
            
            # Read
            iq_in = read_complex_iq(filename, dtype='complex128')
            
            # Verify
            assert np.array_equal(iq_in, iq_out)
        finally:
            os.unlink(filename)
    
    def test_write_read_large_array(self):
        """Test writing and reading large IQ array"""
        # Create test data
        n = 10000
        iq_out = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            # Write
            write_complex_iq(filename, iq_out)
            
            # Read
            iq_in = read_complex_iq(filename)
            
            # Verify
            assert np.allclose(iq_in, iq_out)
        finally:
            os.unlink(filename)
    
    def test_read_partial(self):
        """Test reading partial IQ samples"""
        # Create test data
        iq_out = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j], dtype=np.complex64)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            # Write
            write_complex_iq(filename, iq_out)
            
            # Read only first 3 samples
            iq_in = read_complex_iq(filename, count=3)
            
            # Verify
            assert len(iq_in) == 3
            assert np.array_equal(iq_in, iq_out[:3])
        finally:
            os.unlink(filename)
    
    def test_invalid_dtype(self):
        """Test that invalid dtype raises ValueError"""
        iq = np.array([1+1j, 2+2j], dtype=np.complex64)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            with pytest.raises(ValueError):
                write_complex_iq(filename, iq, dtype='invalid')
            
            with pytest.raises(ValueError):
                read_complex_iq(filename, dtype='invalid')
        finally:
            os.unlink(filename)


class TestInterleavedFloat:
    """Test interleaved float I/O"""
    
    def test_write_interleaved(self):
        """Test writing interleaved float format"""
        iq = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            # Write
            write_iq_interleaved_float(filename, iq)
            
            # Read as float32 array
            data = np.fromfile(filename, dtype=np.float32)
            
            # Should be interleaved: I,Q,I,Q,I,Q
            assert len(data) == 6
            assert data[0] == 1.0  # I of first sample
            assert data[1] == 2.0  # Q of first sample
            assert data[2] == 3.0  # I of second sample
            assert data[3] == 4.0  # Q of second sample
        finally:
            os.unlink(filename)


class TestSampleInfo:
    """Test sample info function"""
    
    def test_get_sample_info_complex64(self):
        """Test getting info for complex64 file"""
        iq = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            write_complex_iq(filename, iq)
            info = get_sample_info(filename, dtype='complex64')
            
            assert info['num_samples'] == 4
            assert info['bytes_per_sample'] == 8
            assert info['file_size_bytes'] == 32
            assert info['dtype'] == 'complex64'
        finally:
            os.unlink(filename)
    
    def test_get_sample_info_complex128(self):
        """Test getting info for complex128 file"""
        iq = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        try:
            write_complex_iq(filename, iq, dtype='complex128')
            info = get_sample_info(filename, dtype='complex128')
            
            assert info['num_samples'] == 3
            assert info['bytes_per_sample'] == 16
            assert info['file_size_bytes'] == 48
            assert info['dtype'] == 'complex128'
        finally:
            os.unlink(filename)
