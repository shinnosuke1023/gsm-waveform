#!/usr/bin/env python3
"""Example: Demodulate GSM BCCH Waveform

This script demonstrates how to read a BCCH waveform file,
demodulate it, and decode the information.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import (
    read_complex_iq,
    gmsk_demodulate,
    detect_burst_by_tsc,
    detect_sch_burst,
    extract_burst_data_114,
    extract_sch_data,
    decode_bcch_pipeline,
    decode_sch_39bits,
    FS_GEN,
    OSR_DEFAULT
)


def demodulate_bcch_file(filename: str, tsc_index: int = 0):
    """Demodulate and decode a BCCH waveform file.
    
    Args:
        filename: Input cfile name
        tsc_index: Expected training sequence code (0-7)
    """
    print("=" * 60)
    print("GSM BCCH Waveform Demodulator")
    print("=" * 60)
    print()
    
    # 1. Read IQ samples
    print(f"Reading IQ samples from '{filename}'...")
    try:
        iq_samples = read_complex_iq(filename)
        print(f"Read {len(iq_samples)} IQ samples")
        print(f"Duration: {len(iq_samples) / FS_GEN * 1000:.2f} ms")
        print()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Please run 'python examples/generate_bcch_waveform.py' first to create a waveform file.")
        return
    
    # 2. Demodulate to bits
    print("Demodulating IQ samples to bits...")
    demod_bits = gmsk_demodulate(iq_samples, osr=OSR_DEFAULT)
    print(f"Demodulated to {len(demod_bits)} bits")
    print()
    
    # 3. Detect SCH burst
    print("Detecting SCH burst...")
    sch_positions = detect_sch_burst(demod_bits, threshold=0.5)
    print(f"Found {len(sch_positions)} SCH burst(s) at positions: {sch_positions}")
    
    if sch_positions:
        # Try to decode SCH
        sch_pos = sch_positions[0]
        # SCH structure: TAIL(3) | DATA(39) | TRAINING(64) | DATA(39) | TAIL(3)
        # Position is where training starts, so data starts 3 bits before
        if sch_pos >= 42:
            sch_burst = demod_bits[sch_pos-42:sch_pos+106]
            if len(sch_burst) >= 148:
                left_data, right_data = extract_sch_data(sch_burst)
                # SCH has same data on both sides
                sch39 = left_data
                bsic, fn, valid = decode_sch_39bits(sch39)
                
                if valid:
                    print(f"  SCH decoded successfully!")
                    print(f"  BSIC: {bsic}")
                    print(f"  Frame Number: {fn}")
                else:
                    print(f"  SCH CRC check failed")
    print()
    
    # 4. Detect normal bursts by TSC
    print(f"Detecting normal bursts with TSC {tsc_index}...")
    burst_positions = detect_burst_by_tsc(demod_bits, tsc_index=tsc_index, threshold=0.6)
    print(f"Found {len(burst_positions)} burst(s) at positions: {burst_positions[:10]}...")
    print()
    
    if len(burst_positions) >= 4:
        # Extract data from first 4 bursts
        print("Extracting data from first 4 bursts...")
        data_bursts = []
        
        for i, pos in enumerate(burst_positions[:4]):
            # TSC position is in the middle of the burst
            # Burst structure: TAIL(3) | DATA(57) | S(1) | TSC(26) | S(1) | DATA(57) | TAIL(3)
            # Position is where TSC starts, so burst starts 61 bits before
            burst_start = pos - 61
            
            if burst_start >= 0 and burst_start + 148 <= len(demod_bits):
                burst = demod_bits[burst_start:burst_start + 148]
                data114 = extract_burst_data_114(burst)
                data_bursts.append(data114)
                print(f"  Burst {i+1}: extracted 114 data bits")
        
        if len(data_bursts) == 4:
            # 5. Decode BCCH
            print()
            print("Decoding BCCH information...")
            info_bits, valid = decode_bcch_pipeline(data_bursts)
            
            if valid:
                print(f"  ✓ BCCH decoded successfully!")
                print(f"  Information bits (184): {len(info_bits)}")
                print(f"  First 32 bits: {info_bits[:32]}")
                print(f"  Parity check: PASSED")
            else:
                print(f"  ✗ BCCH decoding failed (parity check failed)")
                print(f"  This may be due to:")
                print(f"    - Noise in the signal")
                print(f"    - Incorrect burst alignment")
                print(f"    - Wrong TSC index")
        else:
            print(f"Could not extract 4 complete bursts")
    else:
        print(f"Not enough bursts detected for BCCH decoding (need at least 4)")
    
    print()
    print("=" * 60)
    print("Demodulation complete!")
    print("=" * 60)
    print()


def main():
    """Main function"""
    # Default file to read
    input_file = "bcch_waveform.cfile"
    
    # Check if file is provided as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # TSC index (can be provided as second argument)
    tsc_index = 0
    if len(sys.argv) > 2:
        tsc_index = int(sys.argv[2])
    
    demodulate_bcch_file(input_file, tsc_index)


if __name__ == "__main__":
    main()
