#!/usr/bin/env python3
"""End-to-end test: Generate, modulate, demodulate, and decode"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import (
    make_bcch_encoded_456,
    interleave_456_to_4x114,
    build_normal_burst,
    get_tsc,
    modulate_burst_sequence,
    gmsk_demodulate,
    extract_burst_data_114,
    decode_bcch_pipeline,
    OSR_DEFAULT
)


def test_end_to_end():
    """Test complete encode/modulate/demodulate/decode pipeline"""
    print("=" * 60)
    print("End-to-End Test: Encode → Modulate → Demodulate → Decode")
    print("=" * 60)
    print()
    
    # 1. Create known information
    print("Creating test information (184 bits)...")
    info_bits = np.random.randint(0, 2, 184, dtype=np.uint8)
    print(f"First 32 bits: {info_bits[:32]}")
    print()
    
    # 2. Encode
    print("Encoding...")
    encoded_456 = make_bcch_encoded_456(info_bits)
    print(f"Encoded to {len(encoded_456)} bits")
    
    # 3. Interleave
    print("Interleaving to 4 bursts...")
    data_bursts_orig = interleave_456_to_4x114(encoded_456)
    
    # 4. Build normal bursts
    print("Building normal bursts...")
    tsc = get_tsc(0)
    burst_bits = [build_normal_burst(data, tsc) for data in data_bursts_orig]
    print(f"Built {len(burst_bits)} bursts of {len(burst_bits[0])} bits each")
    print()
    
    # 5. Modulate (without guard periods for simplicity)
    print("Modulating to IQ samples...")
    iq = modulate_burst_sequence(burst_bits, guard_samples=0)
    print(f"Generated {len(iq)} IQ samples")
    print()
    
    # 6. Demodulate - try all phase alignments
    print("Demodulating (trying all phase alignments)...")
    all_phases = gmsk_demodulate(iq, osr=OSR_DEFAULT, return_all_phases=True)
    print(f"Got {len(all_phases)} phase alignments, each with {len(all_phases[0])} bits")
    
    # Expected bits per burst
    bits_per_burst = 148
    expected_total_bits = len(burst_bits) * bits_per_burst
    print(f"Expected total bits: {expected_total_bits}")
    print()
    
    # 7. Extract bursts from demodulated data - try all phase alignments
    print("Extracting bursts from demodulated data...")
    
    best_similarity = 0
    best_phase = 0
    best_offset = 0
    best_decoded_info = None
    best_parity_valid = False
    
    # Try each phase alignment
    for phase_idx in range(len(all_phases)):
        demod_bits = all_phases[phase_idx]
        
        # Try different starting positions within this phase
        for offset in range(-10, 11):
            start = offset
            if start < 0 or start + expected_total_bits > len(demod_bits):
                continue
            
            # Extract bursts at this offset
            data_bursts_demod = []
            valid = True
            
            for i in range(4):
                burst_start = start + i * bits_per_burst
                burst_end = burst_start + bits_per_burst
                
                if burst_end > len(demod_bits):
                    valid = False
                    break
                
                burst = demod_bits[burst_start:burst_end]
                data_114 = extract_burst_data_114(burst)
                data_bursts_demod.append(data_114)
            
            if valid and len(data_bursts_demod) == 4:
                # Try decoding
                try:
                    decoded_info, parity_valid = decode_bcch_pipeline(data_bursts_demod)
                    
                    if parity_valid:
                        similarity = np.mean(decoded_info == info_bits)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_phase = phase_idx
                            best_offset = offset
                            best_decoded_info = decoded_info
                            best_parity_valid = parity_valid
                            
                            print(f"  Phase {phase_idx}, offset {offset:+3d}: Parity valid, similarity {similarity:.1%}")
                            
                            if similarity > 0.99:
                                break
                except:
                    pass
        
        if best_similarity > 0.99:
            break
    
    print()
    print(f"Best: Phase {best_phase}, offset {best_offset}, similarity {best_similarity:.1%}")
    print()
    
    # Use best result
    if best_similarity > 0:
        print("Decoding results:")
        print(f"Parity check: {'PASSED' if best_parity_valid else 'FAILED'}")
        
        # Compare with original
        bit_errors = np.sum(best_decoded_info != info_bits)
        
        print(f"Bit errors: {bit_errors} / 184")
        print(f"Similarity: {best_similarity:.1%}")
        print()
        
        if best_similarity > 0.95:
            print("✓ End-to-end test PASSED!")
        else:
            print("✗ End-to-end test FAILED")
            print(f"Original first 32 bits: {info_bits[:32]}")
            print(f"Decoded  first 32 bits: {best_decoded_info[:32]}")
    else:
        print("✗ Could not find valid decoding at any phase/offset")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    test_end_to_end()
