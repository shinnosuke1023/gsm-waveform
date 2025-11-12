"""GSM GMSK Modulator

Implements Gaussian Minimum Shift Keying (GMSK) modulation
with BT=0.3 according to GSM specifications.
"""

import numpy as np
from scipy import signal
from .constants import SYM_RATE, OSR_DEFAULT, BT, GAUSSIAN_FILTER_SPAN, FS_GEN


def design_gaussian_filter(bt: float, osr: int, span: int) -> np.ndarray:
    """Design a Gaussian pulse shaping filter.
    
    Args:
        bt: Bandwidth-Time product (typically 0.3 for GSM)
        osr: Oversampling ratio (samples per symbol)
        span: Filter span in symbols
        
    Returns:
        Gaussian filter coefficients (normalized)
    """
    # Number of taps (must be odd for symmetry)
    ntaps = span * osr + 1
    
    # Time axis in symbol periods (use linspace for exact number of samples)
    t = np.linspace(-span/2, span/2, ntaps)
    
    # Gaussian pulse shape: exp(-α²t²) where α = sqrt(ln(2))/(BT)
    alpha = np.sqrt(np.log(2)) / bt
    h = np.exp(-alpha**2 * t**2)
    
    # Normalize to unit energy
    h = h / np.sum(h)
    
    return h


def gmsk_modulate(bitseq: np.ndarray,
                  fs: float = FS_GEN,
                  osr: int = OSR_DEFAULT,
                  bt: float = BT,
                  span: int = GAUSSIAN_FILTER_SPAN) -> np.ndarray:
    """GMSK modulate a bit sequence.
    
    Process:
    1. Map bits to NRZ: 0 → -1, 1 → +1
    2. Upsample by oversampling ratio
    3. Apply Gaussian pulse shaping filter
    4. Integrate to get phase (MSK relationship)
    5. Generate complex IQ: exp(j*phase)
    
    Args:
        bitseq: numpy array of bits (0/1)
        fs: Sample rate (default: FS_GEN)
        osr: Oversampling ratio (default: OSR_DEFAULT)
        bt: BT parameter (default: BT)
        span: Filter span in symbols (default: GAUSSIAN_FILTER_SPAN)
        
    Returns:
        Complex IQ samples (complex64)
    """
    # 1. NRZ mapping: 0 → -1, 1 → +1
    nrz = 2 * bitseq.astype(np.float64) - 1
    
    # 2. Upsample by inserting zeros
    upsampled = np.zeros(len(nrz) * osr, dtype=np.float64)
    upsampled[::osr] = nrz
    
    # 3. Gaussian filter
    h = design_gaussian_filter(bt, osr, span)
    shaped = np.convolve(upsampled, h, mode='same')
    
    # 4. Integrate to get phase
    # Phase modulation index h=0.5 for MSK
    # phase(t) = π/2 * integral of shaped signal
    phase = np.cumsum(shaped) * (np.pi / 2)
    
    # 5. Complex IQ generation
    iq = np.exp(1j * phase)
    
    return iq.astype(np.complex64)


def modulate_burst_sequence(bursts: list,
                            guard_samples: int = 0,
                            fs: float = FS_GEN,
                            osr: int = OSR_DEFAULT) -> np.ndarray:
    """Modulate a sequence of bursts with guard periods.
    
    Args:
        bursts: List of bit arrays (each burst)
        guard_samples: Number of zero samples between bursts
        fs: Sample rate
        osr: Oversampling ratio
        
    Returns:
        Complex IQ samples for all bursts
    """
    iq_segments = []
    guard = np.zeros(guard_samples, dtype=np.complex64)
    
    for burst in bursts:
        iq = gmsk_modulate(burst, fs=fs, osr=osr)
        iq_segments.append(iq)
        if guard_samples > 0:
            iq_segments.append(guard)
    
    # Concatenate all segments
    full_iq = np.concatenate(iq_segments)
    return full_iq
