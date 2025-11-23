"""GSM GMSK Demodulator

Implements GMSK demodulation and burst detection
for GSM broadcast channels.
"""

import numpy as np
from typing import Tuple, List, Optional
from .constants import (
    OSR_DEFAULT, GAUSSIAN_FILTER_SPAN,
    TSC_TABLE, SCH_TRAINING_SEQ
)


def gmsk_demodulate(iq: np.ndarray, 
                   osr: int = OSR_DEFAULT,
                   return_all_phases: bool = False) -> np.ndarray:
    """GMSK demodulate IQ samples to bit sequence.
    
    Uses frequency discriminator approach:
    1. Compute instantaneous phase
    2. Differentiate to get frequency
    3. Sample at symbol rate
    4. Threshold to recover bits
    
    Args:
        iq: Complex IQ samples
        osr: Oversampling ratio (samples per symbol)
        return_all_phases: If True, returns multiple phase-shifted versions
        
    Returns:
        Demodulated bit sequence (0/1)
        If return_all_phases=True, returns array with shape (osr, n_bits)
    """
    # Ensure complex type
    iq = iq.astype(np.complex128)
    
    # Compute phase difference (frequency discriminator)
    # phase_diff = angle(s[n] * conj(s[n-1]))
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    
    # Normalize by sample period to get frequency deviation
    # For GMSK with h=0.5, frequency deviation is ±0.25 * symbol_rate
    # Phase difference per sample should be ±π/2/osr for ±1 bits
    normalized = phase_diff * osr / (np.pi / 2)
    
    if return_all_phases:
        # Return all possible phase alignments
        phases = []
        for phase_offset in range(osr):
            start_idx = phase_offset
            if start_idx < len(normalized):
                symbols = normalized[start_idx::osr]
                bits = (symbols > 0).astype(np.uint8)
                phases.append(bits)
        
        # Pad to same length
        max_len = max(len(p) for p in phases)
        for i in range(len(phases)):
            if len(phases[i]) < max_len:
                phases[i] = np.pad(phases[i], (0, max_len - len(phases[i])), 'constant')
        
        return np.array(phases)
    
    # Single best phase
    # Start at sample corresponding to first symbol center
    # The Gaussian filter with mode='same' in modulator centers the output
    # but due to upsampling and filter characteristics, the optimal phase
    # is offset by approximately (span/2 * osr + osr // 2 + 1) samples
    delay_symbols = GAUSSIAN_FILTER_SPAN // 2
    delay_samples = delay_symbols * osr
    
    # Optimal phase offset is delay_samples + osr // 2 + 1
    # This accounts for the convolution mode='same' and upsampling alignment
    start_idx = delay_samples + osr // 2 + 1
    
    # Make sure we don't go out of bounds
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
    # Burst structure: TAIL(3) | DATA(57) | S(1) | TSC(26) | S(1) | DATA(57) | TAIL(3)
    # Positions:        0-2      3-59       60     61-86      87     88-144     145-147
    left_data = burst_bits[3:60]    # Positions 3-59: 57 data bits
    right_data = burst_bits[88:145]  # Positions 88-144: 57 data bits (skip TSC + 2 stealing bits)
    
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
