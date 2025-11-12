"""IQ Waveform File I/O Utilities

Functions for reading and writing complex IQ samples to files
in various formats compatible with GNU Radio and SDR tools.
"""

import numpy as np
from typing import Union


def write_complex_iq(filename: str, 
                     iq: np.ndarray,
                     dtype: str = 'complex64') -> None:
    """Write complex IQ samples to file.
    
    Writes IQ samples in interleaved I/Q format (I,Q,I,Q,...) as float32 or float64.
    This format is compatible with GNU Radio File Source and HackRF tools.
    
    Args:
        filename: Output file path
        iq: Complex IQ samples (numpy array)
        dtype: Data type - 'complex64' (default) or 'complex128'
    """
    if dtype == 'complex64':
        iq = iq.astype(np.complex64)
        # Write as raw bytes (I and Q as float32)
        iq.tofile(filename)
    elif dtype == 'complex128':
        iq = iq.astype(np.complex128)
        # Write as raw bytes (I and Q as float64)
        iq.tofile(filename)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'complex64' or 'complex128'")


def read_complex_iq(filename: str, 
                    dtype: str = 'complex64',
                    count: int = -1) -> np.ndarray:
    """Read complex IQ samples from file.
    
    Reads IQ samples in interleaved I/Q format.
    
    Args:
        filename: Input file path
        dtype: Data type - 'complex64' (default) or 'complex128'
        count: Number of samples to read (-1 for all)
        
    Returns:
        Complex IQ samples (numpy array)
    """
    if dtype == 'complex64':
        np_dtype = np.complex64
    elif dtype == 'complex128':
        np_dtype = np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'complex64' or 'complex128'")
    
    iq = np.fromfile(filename, dtype=np_dtype, count=count)
    return iq


def write_iq_interleaved_float(filename: str,
                               iq: np.ndarray) -> None:
    """Write IQ samples as interleaved float32 (I,Q,I,Q,...).
    
    This is an alternative format that explicitly separates I and Q components.
    
    Args:
        filename: Output file path
        iq: Complex IQ samples (numpy array)
    """
    # Separate I and Q components
    i_samples = np.real(iq).astype(np.float32)
    q_samples = np.imag(iq).astype(np.float32)
    
    # Interleave I and Q
    interleaved = np.empty(len(iq) * 2, dtype=np.float32)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples
    
    # Write to file
    interleaved.tofile(filename)


def get_sample_info(filename: str, dtype: str = 'complex64') -> dict:
    """Get information about an IQ file.
    
    Args:
        filename: Input file path
        dtype: Data type - 'complex64' (default) or 'complex128'
        
    Returns:
        Dictionary with file information
    """
    import os
    
    if dtype == 'complex64':
        bytes_per_sample = 8  # 4 bytes I + 4 bytes Q
    elif dtype == 'complex128':
        bytes_per_sample = 16  # 8 bytes I + 8 bytes Q
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    file_size = os.path.getsize(filename)
    num_samples = file_size // bytes_per_sample
    
    return {
        'filename': filename,
        'file_size_bytes': file_size,
        'num_samples': num_samples,
        'dtype': dtype,
        'bytes_per_sample': bytes_per_sample
    }
