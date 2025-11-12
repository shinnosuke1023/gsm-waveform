"""GSM Block Interleaving

Implements block interleaving for BCCH and other control channels
according to GSM 05.03 specifications.
"""

import numpy as np
from typing import List


def interleave_456_to_4x114(bits456: np.ndarray) -> List[np.ndarray]:
    """Block interleaving: 456 bits → [114, 114, 114, 114].
    
    456 bits are arranged column-wise (4 rows × 114 columns).
    Each row forms one burst payload.
    
    According to GSM 05.03, bits are written column-wise and read row-wise.
    
    Args:
        bits456: numpy array of 456 bits
        
    Returns:
        List of 4 numpy arrays, each containing 114 bits
    """
    assert isinstance(bits456, np.ndarray)
    assert bits456.size == 456, f"Expected 456 bits, got {bits456.size}"
    
    M = 4    # Number of bursts
    N = 114  # Bits per burst
    mat = np.zeros((M, N), dtype=np.uint8)
    
    # Write column-wise (GSM standard interleaving)
    for k in range(456):
        r = k % M
        c = k // M
        mat[r, c] = bits456[k]
    
    # Read row-wise
    chunks = [mat[r, :].copy() for r in range(M)]
    return chunks
