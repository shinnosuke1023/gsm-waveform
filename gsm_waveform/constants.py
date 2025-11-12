"""GSM Waveform Generation Constants

All constants defined according to GSM 05.02 and 05.03 specifications.
"""

import numpy as np

# Sampling / Symbol rates
SYM_RATE = 270833.3333333333   # symbols/sec (GSM bit rate)
OSR_DEFAULT = 8                # Base oversampling ratio

# GMSK / Gaussian filter parameters
BT = 0.3                       # Gaussian filter BT parameter
GAUSSIAN_FILTER_SPAN = 4       # Filter length in symbols

# Convolutional code parameters (rate=1/2, K=5)
CONV_K = 5
CONV_GENERATORS = (0o23, 0o35)  # Octal: G0=0b10011, G1=0b11101

# BCCH encoding parameters
BCCH_INFO_BITS = 184
BCCH_FIRE_PARITY = 40          # FIRE code adds 40 parity bits
BCCH_AFTER_BLOCK = BCCH_INFO_BITS + BCCH_FIRE_PARITY  # 224
BCCH_TAIL_BITS = 4              # Tail bits before convolutional encoding
BCCH_AFTER_CONV = 456          # After 1/2 rate convolutional encoding: (224+4)*2

# BCCH interleaving parameters
BCCH_BURSTS = 4
BCCH_BITS_PER_BURST = 114

# Burst structure parameters
TAIL_LEN = 3
TRAINING_LEN_NB = 26           # Normal Burst training sequence length
GUARD_LEN = 8.25               # Guard period in bit periods

# System parameters
MF_PERIOD = 51                 # Downlink multiframe length

# Sample rate for generation
FS_GEN = SYM_RATE * OSR_DEFAULT  # ~2.167 MHz

# Training Sequence Codes (TSC 0-7) - 26 bits each
# Defined in GSM 05.02 Section 5.2.3
TSC_TABLE = [
    np.array([0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1], dtype=np.uint8),  # TSC 0
    np.array([0,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1], dtype=np.uint8),  # TSC 1
    np.array([0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,1,1,1,0], dtype=np.uint8),  # TSC 2
    np.array([0,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0], dtype=np.uint8),  # TSC 3
    np.array([0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1], dtype=np.uint8),  # TSC 4
    np.array([0,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,0], dtype=np.uint8),  # TSC 5
    np.array([1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,1], dtype=np.uint8),  # TSC 6
    np.array([1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0], dtype=np.uint8),  # TSC 7
]

# SCH extended training sequence (64 bits)
# Defined in GSM 05.02 Section 5.2.5
SCH_TRAINING_SEQ = np.array([
    1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,
    0,0,1,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1
], dtype=np.uint8)

# CRC polynomial for SCH
SCH_CRC_POLY = 0b10110111101  # D^10 + D^8 + D^6 + D^5 + D^4 + D^2 + 1
