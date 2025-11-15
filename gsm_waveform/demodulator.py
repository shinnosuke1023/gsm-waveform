"""GSM GMSK Demodulator

Implements GMSK demodulation and burst detection
for GSM broadcast channels.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional
from .constants import (
    SYM_RATE, OSR_DEFAULT, BT, GAUSSIAN_FILTER_SPAN,
    TSC_TABLE, SCH_TRAINING_SEQ, TRAINING_LEN_NB
)


def gmsk_demodulate(iq: np.ndarray, 
                   osr: int = OSR_DEFAULT) -> np.ndarray:
    """GMSK demodulate IQ samples to bit sequence.
    
    Uses frequency discriminator approach:
    1. Compute instantaneous phase
    2. Differentiate to get frequency
    3. Sample at symbol rate
    4. Threshold to recover bits
    
    Args:
        iq: Complex IQ samples
        osr: Oversampling ratio (samples per symbol)
        
    Returns:
        Demodulated bit sequence (0/1)
    """
    # Ensure complex type
    iq = iq.astype(np.complex128)
    
    # Compute phase difference (frequency discriminator)
    # phase_diff = angle(s[n] * conj(s[n-1]))
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    
    # Apply low-pass filtering to smooth out noise
    # Use a simple moving average
    if len(phase_diff) > osr:
        window = np.ones(osr) / osr
        phase_diff_filtered = np.convolve(phase_diff, window, mode='same')
    else:
        phase_diff_filtered = phase_diff
    
    # Normalize by sample period to get frequency deviation
    # For GMSK with h=0.5, frequency deviation is ±0.25 * symbol_rate
    # Phase difference per sample should be ±π/2/osr for ±1 bits
    normalized = phase_diff_filtered * osr / (np.pi / 2)
    
    # Downsample to symbol rate (take samples at symbol centers)
    # Account for filter delay and sample at appropriate points
    # Add delay compensation for the Gaussian filter used in modulation
    delay = GAUSSIAN_FILTER_SPAN * osr // 2
    start_idx = delay
    
    # Ensure we don't go out of bounds
    if start_idx >= len(normalized):
        start_idx = osr // 2
    
    symbols = normalized[start_idx::osr]
    
    # Threshold: positive → 1, negative → 0
    bits = (symbols > 0).astype(np.uint8)
    
    return bits


def correlate_sequence(signal_bits: np.ndarray, 
                       reference: np.ndarray) -> np.ndarray:
    """Compute cross-correlation between signal and reference sequence.
    
    Args:
        signal_bits: Input bit sequence (0/1)
        reference: Reference bit sequence (0/1)
        
    Returns:
        Correlation values at each position
    """
    # Convert bits to ±1 for correlation
    sig = 2 * signal_bits.astype(np.float64) - 1
    ref = 2 * reference.astype(np.float64) - 1
    
    # Compute correlation
    corr = np.correlate(sig, ref, mode='valid')
    
    return corr


def detect_burst_by_tsc(demod_bits: np.ndarray,
                       tsc_index: int = 0,
                       threshold: float = 0.7) -> List[int]:
    """Detect burst positions by correlating with TSC.
    
    Args:
        demod_bits: Demodulated bit sequence
        tsc_index: Training sequence code index (0-7)
        threshold: Correlation threshold (0-1)
        
    Returns:
        List of detected burst start positions
    """
    tsc = TSC_TABLE[tsc_index]
    corr = correlate_sequence(demod_bits, tsc)
    
    # Normalize correlation
    max_corr = len(tsc)
    norm_corr = corr / max_corr
    
    # Find peaks above threshold
    peaks = np.where(norm_corr >= threshold)[0]
    
    # Filter peaks to avoid duplicates (keep local maxima)
    if len(peaks) == 0:
        return []
    
    burst_positions = []
    min_spacing = 100  # Minimum bits between bursts
    
    for peak in peaks:
        # Check if this peak is far enough from previous detections
        if len(burst_positions) == 0 or peak - burst_positions[-1] >= min_spacing:
            burst_positions.append(int(peak))
    
    return burst_positions


def extract_normal_burst_data(burst_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract data bits from a normal burst.
    
    Normal burst structure: TAIL(3) | DATA(57) | S | TSC(26) | S | DATA(57) | TAIL(3)
    
    Args:
        burst_bits: 148-bit normal burst
        
    Returns:
        Tuple of (left_57_bits, right_57_bits)
    """
    assert len(burst_bits) >= 148, f"Expected at least 148 bits, got {len(burst_bits)}"
    
    # Extract data fields (skip tail and stealing bits)
    left_data = burst_bits[3:60]    # Skip 3 tail bits, take 57 data bits
    right_data = burst_bits[87:144]  # Skip TSC (26) + stealing (2), take 57 data bits
    
    return left_data, right_data


def extract_burst_data_114(burst_bits: np.ndarray) -> np.ndarray:
    """Extract 114 data bits from a normal burst.
    
    Args:
        burst_bits: Normal burst (148 bits)
        
    Returns:
        114 data bits (57 + 57)
    """
    left, right = extract_normal_burst_data(burst_bits)
    return np.concatenate([left, right])


def detect_fcch_burst(iq: np.ndarray,
                     osr: int = OSR_DEFAULT) -> Optional[int]:
    """Detect FCCH burst by finding continuous tone.
    
    FCCH produces a pure tone at fc + 67.7 kHz.
    
    Args:
        iq: Complex IQ samples
        osr: Oversampling ratio
        
    Returns:
        Start position of FCCH burst, or None if not found
    """
    # Compute instantaneous frequency
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    
    # FCCH has constant frequency (all zeros → constant phase rate)
    # Look for segments with very stable frequency
    window_size = 148 * osr  # One burst length
    
    # Compute frequency variance in sliding windows
    variances = []
    for i in range(len(phase_diff) - window_size):
        window = phase_diff[i:i+window_size]
        variances.append(np.var(window))
    
    if len(variances) == 0:
        return None
    
    # Find minimum variance (most stable frequency)
    min_idx = np.argmin(variances)
    
    # Check if variance is low enough
    if variances[min_idx] < 0.01:  # Threshold for "stable"
        return min_idx
    
    return None


def detect_sch_burst(demod_bits: np.ndarray,
                    threshold: float = 0.6) -> List[int]:
    """Detect SCH burst by correlating with extended training sequence.
    
    Args:
        demod_bits: Demodulated bit sequence
        threshold: Correlation threshold (0-1)
        
    Returns:
        List of detected SCH burst positions
    """
    corr = correlate_sequence(demod_bits, SCH_TRAINING_SEQ)
    
    # Normalize correlation
    max_corr = len(SCH_TRAINING_SEQ)
    norm_corr = corr / max_corr
    
    # Find peaks above threshold
    peaks = np.where(norm_corr >= threshold)[0]
    
    # Filter for local maxima with minimum spacing
    burst_positions = []
    min_spacing = 100
    
    for peak in peaks:
        if len(burst_positions) == 0 or peak - burst_positions[-1] >= min_spacing:
            burst_positions.append(int(peak))
    
    return burst_positions


def extract_sch_data(burst_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract data from SCH burst.
    
    SCH structure: TAIL(3) | DATA(39) | EXTENDED_TSC(64) | DATA(39) | TAIL(3)
    
    Args:
        burst_bits: 148-bit SCH burst
        
    Returns:
        Tuple of (left_39_bits, right_39_bits)
    """
    assert len(burst_bits) >= 148, f"Expected at least 148 bits, got {len(burst_bits)}"
    
    left_data = burst_bits[3:42]     # Skip 3 tail, take 39 data
    right_data = burst_bits[106:145] # Skip 64-bit TSC, take 39 data
    
    return left_data, right_data
