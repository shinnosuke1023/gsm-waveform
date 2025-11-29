#!/usr/bin/env python3
"""End-to-end test for System Information encoding/decoding

This script demonstrates encoding System Information Type 3,
generating a complete BCCH waveform, demodulating it, and
decoding the System Information.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import (
    encode_system_information_type3,
    decode_system_information_type3,
    format_system_information,
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


def test_system_information_e2e():
    """Test System Information through complete encode/modulate/demodulate/decode pipeline"""
    print("=" * 60)
    print("System Information End-to-End Test")
    print("=" * 60)
    print()
    
    # 1. Create System Information Type 3 message
    print("Creating System Information Type 3 message (GSM 04.08)...")
    
    cell_identity = 12345
    location_area_code = 100
    mobile_country_code = 440
    mobile_network_code = 10
    arfcn = 975
    neighbor_cells = [980, 985]  # Max 2 neighbors due to 184-bit constraint
    
    print(f"  Cell Identity: {cell_identity}")
    print(f"  Location Area Code: {location_area_code}")
    print(f"  MCC: {mobile_country_code}, MNC: {mobile_network_code}")
    print(f"  ARFCN: {arfcn}")
    print(f"  Neighbor Cells: {neighbor_cells}")
    print()
    
    # Encode System Information
    info_bits = encode_system_information_type3(
        cell_identity=cell_identity,
        location_area_code=location_area_code,
        mobile_country_code=mobile_country_code,
        mobile_network_code=mobile_network_code,
        arfcn=arfcn,
        neighbor_cells=neighbor_cells
    )
    
    # 2. Encode for BCCH
    print("Encoding for BCCH...")
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
    print(f"Got {len(all_phases)} phase alignments")
    print()
    
    # Expected bits per burst
    bits_per_burst = 148
    expected_total_bits = len(burst_bits) * bits_per_burst
    
    # 7. Extract bursts from demodulated data - try all phase alignments
    print("Extracting bursts and decoding...")
    
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
                            
                            if similarity > 0.99:
                                break
                except Exception:
                    # Decoding may fail for some offsets due to invalid alignment
                    pass
        
        if best_similarity > 0.99:
            break
    
    print(f"Best alignment: Phase {best_phase}, offset {best_offset}")
    print(f"Similarity: {best_similarity:.1%}")
    print()
    
    # Use best result
    if best_similarity > 0:
        print("Decoding results:")
        print(f"Parity check: {'PASSED' if best_parity_valid else 'FAILED'}")
        
        # Compare with original
        bit_errors = np.sum(best_decoded_info != info_bits)
        print(f"Bit errors: {bit_errors} / 184")
        print()
        
        if best_similarity > 0.95:
            print("✓ BCCH decoding PASSED!")
            print()
            
            # Display decoded System Information
            print("Decoding System Information...")
            decoded_si = decode_system_information_type3(best_decoded_info)
            
            print()
            print("Decoded System Information:")
            print(f"  Cell Identity: {decoded_si['cell_identity']}")
            print(f"  Location Area Code: {decoded_si['location_area_code']}")
            print(f"  MCC: {decoded_si['mobile_country_code']}, MNC: {decoded_si['mobile_network_code']}")
            print(f"  Neighbor Cells: {decoded_si['neighbor_cells']}")
            print()
            
            # Verify
            all_match = (
                decoded_si['cell_identity'] == cell_identity and
                decoded_si['location_area_code'] == location_area_code and
                decoded_si['mobile_country_code'] == mobile_country_code and
                decoded_si['mobile_network_code'] == mobile_network_code and
                decoded_si['neighbor_cells'] == neighbor_cells
            )
            
            if all_match:
                print("✓ System Information matches original!")
                print()
                print("=" * 60)
                print("End-to-End Test PASSED!")
                print("=" * 60)
            else:
                print("✗ System Information does not match original")
                print(f"  Expected Cell Identity: {cell_identity}, Got: {decoded_si['cell_identity']}")
                print(f"  Expected LAC: {location_area_code}, Got: {decoded_si['location_area_code']}")
                print(f"  Expected MCC: {mobile_country_code}, Got: {decoded_si['mobile_country_code']}")
                print(f"  Expected MNC: {mobile_network_code}, Got: {decoded_si['mobile_network_code']}")
                print(f"  Expected Neighbors: {neighbor_cells}, Got: {decoded_si['neighbor_cells']}")
        else:
            print("✗ BCCH decoding FAILED")
            print(f"Original first 32 bits: {info_bits[:32]}")
            print(f"Decoded  first 32 bits: {best_decoded_info[:32]}")
    else:
        print("✗ Could not find valid decoding at any phase/offset")
    
    print()


if __name__ == "__main__":
    test_system_information_e2e()
