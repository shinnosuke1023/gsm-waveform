"""GSM Burst Construction

Implements Normal Burst, FCCH, and SCH burst construction
according to GSM 05.02 specifications.
"""

import numpy as np
from typing import Optional
from .constants import (
    TAIL_LEN, TRAINING_LEN_NB, TSC_TABLE, 
    SCH_TRAINING_SEQ, SCH_CRC_POLY
)


def build_normal_burst(data114: np.ndarray, 
                       tsc: np.ndarray,
                       stealing_bits: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a GSM Normal Burst.
    
    Structure: TAIL(3) | DATA(57) | S | TSC(26) | S | DATA(57) | TAIL(3)
    Total: 148 bits (excluding guard period)
    
    Args:
        data114: 114 data bits (will be split into 2×57)
        tsc: 26-bit training sequence
        stealing_bits: Optional 2 stealing bits (default: [0, 0])
        
    Returns:
        numpy array of 148 bits representing the normal burst
    """
    assert data114.size == 114, f"Expected 114 data bits, got {data114.size}"
    assert tsc.size == 26, f"Expected 26 TSC bits, got {tsc.size}"
    
    if stealing_bits is None:
        stealing_bits = np.array([0, 0], dtype=np.uint8)
    
    tail = np.zeros(TAIL_LEN, dtype=np.uint8)
    data1 = data114[:57]
    data2 = data114[57:]
    
    burst = np.concatenate([
        tail,
        data1,
        stealing_bits[:1],
        tsc,
        stealing_bits[1:2],
        data2,
        tail
    ])
    
    assert burst.size == 148
    return burst


def build_fcch_burst() -> np.ndarray:
    """Build a Frequency Correction Channel (FCCH) burst.
    
    FCCH burst is all zeros (tail bits are also zero).
    When GMSK modulated, this produces a continuous tone at 
    fc + 67.7 kHz for frequency synchronization.
    
    Structure: TAIL(3) | FIXED_BITS(142) | TAIL(3)
    All bits are 0.
    
    Returns:
        numpy array of 148 bits (all zeros)
    """
    burst = np.zeros(148, dtype=np.uint8)
    return burst


def compute_sch_crc10(info25: np.ndarray) -> np.ndarray:
    """Compute the 10-bit CRC for SCH information bits.
    
    Polynomial: D^10 + D^8 + D^6 + D^5 + D^4 + D^2 + 1
    
    Args:
        info25: numpy array of 25 bits (BSIC + RFN)
        
    Returns:
        10-bit parity array
    """
    m = 25
    work = np.concatenate([info25.astype(np.uint8), np.zeros(10, dtype=np.uint8)])
    poly = np.array([1,0,1,1,0,1,1,1,0,1,1], dtype=np.uint8)
    
    for i in range(m):
        if work[i] == 1:
            work[i:i+11] ^= poly
    
    remainder = work[m:m+10]
    return remainder.copy()


def build_sch_39bits(bsic: int, fn: int) -> np.ndarray:
    """Build 39-bit SCH information block.
    
    Args:
        bsic: Base Station Identity Code (0-63, 6 bits)
        fn: TDMA frame number
        
    Returns:
        39-bit array: INFO(25) | CRC(10) | TAIL(4)
    """
    # Calculate frame number components
    T1 = (fn // (26 * 51)) % 2048  # 11 bits
    T2 = fn % 26                     # 5 bits
    T3 = fn % 51                     # 6 bits
    T3p = (T3 - 1) // 10 if T3 > 0 else 0  # 3 bits

    # Convert to bit arrays (MSB first)
    bsic_bits = np.array([(bsic >> i) & 1 for i in reversed(range(6))], dtype=np.uint8)
    t1_bits = np.array([(T1 >> i) & 1 for i in reversed(range(11))], dtype=np.uint8)
    t2_bits = np.array([(T2 >> i) & 1 for i in reversed(range(5))], dtype=np.uint8)
    t3p_bits = np.array([(T3p >> i) & 1 for i in reversed(range(3))], dtype=np.uint8)

    # Assemble 25-bit information
    info25 = np.concatenate([bsic_bits, t1_bits, t2_bits, t3p_bits])
    
    # Calculate CRC
    parity10 = compute_sch_crc10(info25)
    
    # Add 4 tail bits
    tail4 = np.zeros(4, dtype=np.uint8)

    sch39 = np.concatenate([info25, parity10, tail4])
    assert sch39.size == 39
    return sch39


def build_sch_burst(bsic: int, fn: int) -> np.ndarray:
    """Build a Synchronization Channel (SCH) burst.
    
    Structure: TAIL(3) | DATA(39) | EXTENDED_TSC(64) | DATA(39) | TAIL(3)
    Total: 148 bits
    
    Args:
        bsic: Base Station Identity Code (0-63)
        fn: TDMA frame number
        
    Returns:
        numpy array of 148 bits representing the SCH burst
    """
    info39 = build_sch_39bits(bsic, fn)
    tail = np.zeros(TAIL_LEN, dtype=np.uint8)
    
    # SCH has extended training sequence (64 bits) in the middle
    # and data split as 39 bits on each side
    burst = np.concatenate([
        tail,
        info39,
        SCH_TRAINING_SEQ,
        info39,  # Repeat same data on both sides
        tail
    ])
    
    assert burst.size == 148
    return burst


def get_tsc(tsc_index: int = 0) -> np.ndarray:
    """Get Training Sequence Code by index.
    
    Args:
        tsc_index: TSC index (0-7)
        
    Returns:
        26-bit training sequence
    """
    assert 0 <= tsc_index <= 7, f"TSC index must be 0-7, got {tsc_index}"
    return TSC_TABLE[tsc_index].copy()
