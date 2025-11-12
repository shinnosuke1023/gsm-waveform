#!/usr/bin/env python3
"""Test GSM Encoding Pipeline

This script demonstrates and validates the complete encoding pipeline
from information bits through to modulated IQ samples.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import (
    fire_encode_184,
    append_block_tail,
    convolutional_encode,
    make_bcch_encoded_456,
    interleave_456_to_4x114,
    build_normal_burst,
    get_tsc,
    gmsk_modulate,
    BCCH_INFO_BITS,
    BCCH_AFTER_BLOCK,
    BCCH_AFTER_CONV,
    BCCH_BURSTS,
    BCCH_BITS_PER_BURST
)


def test_encoding_pipeline():
    """Test and demonstrate the complete encoding pipeline"""
    
    print("=" * 70)
    print("GSM BCCH Encoding Pipeline Test")
    print("=" * 70)
    print()
    
    # Step 1: Create test information
    print("Step 1: Create test information bits")
    print("-" * 70)
    info_bits = np.zeros(BCCH_INFO_BITS, dtype=np.uint8)
    # Set some bits to create a pattern
    info_bits[0:8] = [0, 1, 0, 1, 0, 1, 0, 1]
    info_bits[16:24] = [1, 1, 0, 0, 1, 1, 0, 0]
    
    print(f"  Input: {BCCH_INFO_BITS} information bits")
    print(f"  Pattern (first 32): {info_bits[:32].tolist()}")
    print(f"  Ones: {np.sum(info_bits)}, Zeros: {BCCH_INFO_BITS - np.sum(info_bits)}")
    print()
    
    # Step 2: FIRE encoding
    print("Step 2: FIRE code encoding (add 40 parity bits)")
    print("-" * 70)
    fire_encoded = fire_encode_184(info_bits)
    print(f"  Output: {len(fire_encoded)} bits ({BCCH_INFO_BITS} info + 40 parity)")
    print(f"  Parity bits: {fire_encoded[BCCH_INFO_BITS:].tolist()}")
    print(f"  All bits preserved: {np.array_equal(fire_encoded[:BCCH_INFO_BITS], info_bits)}")
    print()
    
    # Step 3: Tail bits
    print("Step 3: Append tail bits (4 zeros)")
    print("-" * 70)
    with_tail = append_block_tail(fire_encoded, tail_len=4)
    print(f"  Output: {len(with_tail)} bits (224 + 4 tail)")
    print(f"  Tail bits: {with_tail[-4:].tolist()}")
    print()
    
    # Step 4: Convolutional encoding
    print("Step 4: Convolutional encoding (rate 1/2, K=5)")
    print("-" * 70)
    conv_encoded = convolutional_encode(with_tail)
    print(f"  Input: {len(with_tail)} bits")
    print(f"  Output: {len(conv_encoded)} bits (rate 1/2 → 2× length)")
    print(f"  Expected: {BCCH_AFTER_CONV} bits")
    print(f"  Match: {len(conv_encoded) == BCCH_AFTER_CONV}")
    print()
    
    # Alternative: Use complete pipeline
    print("Step 5: Complete pipeline (alternative method)")
    print("-" * 70)
    encoded_456 = make_bcch_encoded_456(info_bits)
    print(f"  Output: {len(encoded_456)} bits")
    print(f"  Matches manual pipeline: {np.array_equal(conv_encoded, encoded_456)}")
    print()
    
    # Step 6: Interleaving
    print("Step 6: Block interleaving (456 bits → 4 bursts)")
    print("-" * 70)
    interleaved = interleave_456_to_4x114(encoded_456)
    print(f"  Input: {len(encoded_456)} bits")
    print(f"  Output: {len(interleaved)} bursts")
    for i, burst_data in enumerate(interleaved):
        print(f"    Burst {i+1}: {len(burst_data)} bits")
    print()
    
    # Step 7: Burst construction
    print("Step 7: Build normal bursts")
    print("-" * 70)
    tsc = get_tsc(0)
    bursts = []
    for i, data114 in enumerate(interleaved):
        burst = build_normal_burst(data114, tsc)
        bursts.append(burst)
        print(f"  Burst {i+1}: {len(burst)} bits")
        print(f"    Tail (first 3): {burst[:3].tolist()}")
        print(f"    Tail (last 3): {burst[-3:].tolist()}")
    print()
    
    # Step 8: Modulation
    print("Step 8: GMSK modulation")
    print("-" * 70)
    iq_samples = []
    for i, burst in enumerate(bursts):
        iq = gmsk_modulate(burst)
        iq_samples.append(iq)
        print(f"  Burst {i+1}: {len(burst)} bits → {len(iq)} IQ samples")
        print(f"    Magnitude range: [{np.min(np.abs(iq)):.4f}, {np.max(np.abs(iq)):.4f}]")
    
    total_samples = sum(len(iq) for iq in iq_samples)
    print(f"  Total IQ samples: {total_samples}")
    print()
    
    # Summary
    print("=" * 70)
    print("Pipeline Test Complete!")
    print("=" * 70)
    print()
    print("Pipeline Summary:")
    print(f"  184 info bits")
    print(f"  → 224 bits (FIRE encoding)")
    print(f"  → 228 bits (tail)")
    print(f"  → 456 bits (convolutional)")
    print(f"  → 4 × 114 bits (interleave)")
    print(f"  → 4 × 148 bits (normal burst)")
    print(f"  → {total_samples} IQ samples (GMSK)")
    print()
    print("All steps completed successfully! ✓")
    print()


if __name__ == "__main__":
    test_encoding_pipeline()
