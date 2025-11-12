"""GSM Waveform Generator

A Python library for generating GSM broadcast channel waveforms (BCCH, FCCH, SCH)
conforming to 3GPP/ETSI GSM specifications (TS 45.002, TS 45.003).
"""

__version__ = "0.1.0"

from .constants import *
from .encoding import fire_encode_184, append_block_tail, convolutional_encode, make_bcch_encoded_456
from .interleave import interleave_456_to_4x114
from .burst_builder import build_normal_burst, build_fcch_burst, build_sch_burst, get_tsc
from .modulator import gmsk_modulate, modulate_burst_sequence
from .wavefile import write_complex_iq, read_complex_iq, get_sample_info

__all__ = [
    # Constants
    'SYM_RATE', 'OSR_DEFAULT', 'BT', 'GAUSSIAN_FILTER_SPAN',
    'CONV_K', 'CONV_GENERATORS',
    'BCCH_INFO_BITS', 'BCCH_FIRE_PARITY', 'BCCH_AFTER_BLOCK', 'BCCH_AFTER_CONV',
    'BCCH_BURSTS', 'BCCH_BITS_PER_BURST',
    'TAIL_LEN', 'TRAINING_LEN_NB', 'GUARD_LEN',
    'MF_PERIOD', 'FS_GEN',
    'TSC_TABLE',
    # Encoding
    'fire_encode_184', 'append_block_tail', 'convolutional_encode', 'make_bcch_encoded_456',
    # Interleaving
    'interleave_456_to_4x114',
    # Burst building
    'build_normal_burst', 'build_fcch_burst', 'build_sch_burst', 'get_tsc',
    # Modulation
    'gmsk_modulate', 'modulate_burst_sequence',
    # File IO
    'write_complex_iq', 'read_complex_iq', 'get_sample_info',
]
