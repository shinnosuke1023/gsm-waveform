"""GSM Channel Encoding

Implements FIRE code (block parity) and convolutional encoding
according to GSM 05.03 specifications.
"""

import numpy as np
from typing import Tuple
from .constants import CONV_K, CONV_GENERATORS


def _bits_to_list(bits: np.ndarray) -> list:
    """Convert bit array to mutable list."""
    return [int(b) for b in bits.tolist()]


def _poly_div_mod(dividend_bits: list, divisor_bits: list) -> list:
    """Compute remainder of dividend_bits / divisor_bits over GF(2).
    
    Args:
        dividend_bits: List of bits (MSB first)
        divisor_bits: List of bits (MSB first)
        
    Returns:
        Remainder bits list (length = len(divisor_bits)-1)
    """
    work = dividend_bits.copy()
    n = len(work)
    m = len(divisor_bits)
    for i in range(n - m + 1):
        if work[i] == 1:
            for j in range(m):
                work[i + j] ^= divisor_bits[j]
    remainder = work[-(m - 1):]
    return remainder


def fire_encode_184(info_bits: np.ndarray) -> np.ndarray:
    """Encode 184 information bits using FIRE code.
    
    FIRE polynomial: g(D) = (D^23 + 1) × (D^17 + D^3 + 1)
    
    Args:
        info_bits: numpy array of 184 bits (0/1), MSB-first ordering
        
    Returns:
        numpy array of 224 bits (info + 40 parity), MSB-first
    """
    assert isinstance(info_bits, np.ndarray)
    assert info_bits.size == 184, f"Expected 184 bits, got {info_bits.size}"

    # Build g(D) = (D^23 + 1) × (D^17 + D^3 + 1)
    # First factor: f1(D) = D^23 + 1
    f1 = [0] * 24
    f1[0] = 1   # D^23 term (MSB first)
    f1[-1] = 1  # D^0 term

    # Second factor: f2(D) = D^17 + D^3 + 1
    f2 = [0] * 18
    f2[0] = 1          # D^17 term
    f2[17 - 3] = 1     # D^3 term
    f2[-1] = 1         # D^0 term

    # Multiply f1 and f2 (polynomial multiplication over GF(2))
    deg_g = 23 + 17  # = 40
    len_g = deg_g + 1
    g = [0] * len_g
    for i, b1 in enumerate(f1):
        if b1:
            for j, b2 in enumerate(f2):
                if b2:
                    idx = i + j
                    g[idx] ^= 1

    # Form dividend = info_bits << 40 (append 40 zeros)
    dividend = _bits_to_list(info_bits) + [0] * 40

    # Compute remainder
    remainder = _poly_div_mod(dividend, g)
    assert len(remainder) == 40

    parity = np.array(remainder, dtype=np.uint8)
    out = np.concatenate([info_bits.astype(np.uint8), parity])
    assert out.size == 224
    return out


def append_block_tail(bits: np.ndarray, tail_len: int = 4) -> np.ndarray:
    """Append tail bits before convolutional encoding.
    
    Args:
        bits: numpy array of bits
        tail_len: Number of tail bits to append (default: 4)
        
    Returns:
        numpy array with tail bits appended
    """
    assert isinstance(bits, np.ndarray)
    tail = np.zeros(tail_len, dtype=np.uint8)
    return np.concatenate([bits.astype(np.uint8), tail])


def convolutional_encode(bits: np.ndarray, 
                        generators: Tuple[int, int] = CONV_GENERATORS,
                        K: int = CONV_K) -> np.ndarray:
    """Convolutional encoder (rate 1/2, constraint length K).
    
    Args:
        bits: numpy array of bits (0/1)
        generators: tuple of octal integers for taps (G0, G1)
        K: constraint length (default: 5)
        
    Returns:
        numpy array of length len(bits)*2 bits
        Output order per input bit: [g0_out, g1_out]
    """
    assert isinstance(bits, np.ndarray)
    n = bits.size
    sr = [0] * (K - 1)  # Shift register
    out = []
    
    for b in bits.tolist():
        b = int(b)
        # Compute outputs for each generator
        for g in generators:
            y = 0
            # Generator bit i corresponds to tap for D^i (i=0 is current input)
            for i in range(K):
                if ((g >> i) & 1):
                    if i == 0:
                        y ^= b
                    else:
                        y ^= sr[i - 1]
            out.append(y)
        # Shift register update: insert input at front
        sr.insert(0, b)
        sr = sr[:(K - 1)]
    
    return np.array(out, dtype=np.uint8)


def make_bcch_encoded_456(info184: np.ndarray) -> np.ndarray:
    """Complete BCCH encoding pipeline: FIRE → tail → convolutional.
    
    Args:
        info184: 184-bit information array
        
    Returns:
        456-bit encoded array ready for interleaving
    """
    assert info184.size == 184
    fire = fire_encode_184(info184)
    block = append_block_tail(fire, tail_len=4)
    conv = convolutional_encode(block, generators=CONV_GENERATORS, K=CONV_K)
    assert conv.size == 456
    return conv
