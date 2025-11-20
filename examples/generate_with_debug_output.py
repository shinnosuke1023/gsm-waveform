#!/usr/bin/env python3
"""Example: Generate GSM BCCH Waveform with Debug Output

This script demonstrates the complete encoding pipeline and outputs
both bit sequences and IQ data for debugging and verification.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import (
    make_bcch_encoded_456,
    interleave_456_to_4x114,
    build_normal_burst,
    build_fcch_burst,
    build_sch_burst,
    get_tsc,
    gmsk_modulate,
    modulate_burst_sequence,
    write_complex_iq,
    FS_GEN
)


def save_bit_sequence(filename: str, bits: np.ndarray, description: str = ""):
    """Save bit sequence to a text file."""
    with open(filename, 'w') as f:
        if description:
            f.write(f"# {description}\n")
            f.write(f"# Length: {len(bits)} bits\n")
            f.write("#" + "=" * 70 + "\n\n")
        
        # Write bits in groups of 8 for readability
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            f.write(''.join(str(b) for b in byte_bits))
            if i + 8 < len(bits):
                f.write(' ')
            if (i + 8) % 64 == 0:
                f.write('\n')
        f.write('\n')


def main():
    """Main function to generate waveform with debug output"""
    
    print("=" * 80)
    print("GSM BCCH Waveform Generator with Debug Output")
    print("=" * 80)
    print()
    
    # Create sample information bits (184 bits)
    print("Step 1: Creating Information Bits (184 bits)")
    print("-" * 80)
    info_bits = np.zeros(184, dtype=np.uint8)
    # Set some bits to create a non-trivial pattern
    info_bits[0:8] = [0, 1, 0, 1, 0, 1, 0, 1]  # Sample pattern
    info_bits[8:16] = [1, 1, 0, 0, 1, 1, 0, 0]  # Another pattern
    
    print(f"Information bits (first 32): {info_bits[:32]}")
    print(f"Total length: {len(info_bits)} bits")
    save_bit_sequence("01_info_bits.txt", info_bits, "Original Information Bits (184)")
    print("✓ Saved to: 01_info_bits.txt")
    print()
    
    # Encode with FIRE code and convolutional encoding
    print("Step 2: Encoding (FIRE + Convolutional)")
    print("-" * 80)
    encoded_456 = make_bcch_encoded_456(info_bits)
    print(f"After FIRE code (184 → 224 bits) + tail (4 bits) + convolutional (rate 1/2)")
    print(f"Encoded length: {len(encoded_456)} bits")
    print(f"Encoded bits (first 32): {encoded_456[:32]}")
    save_bit_sequence("02_encoded_456.txt", encoded_456, 
                     "After FIRE code + tail + convolutional encoding (456)")
    print("✓ Saved to: 02_encoded_456.txt")
    print()
    
    # Interleave to 4 bursts
    print("Step 3: Interleaving to 4 Bursts")
    print("-" * 80)
    data_bursts = interleave_456_to_4x114(encoded_456)
    print(f"Split into {len(data_bursts)} bursts of 114 bits each")
    for i, burst_data in enumerate(data_bursts):
        print(f"  Burst {i+1} data (first 16): {burst_data[:16]}")
        save_bit_sequence(f"03_burst{i+1}_data_114.txt", burst_data,
                         f"Burst {i+1} Data Payload (114 bits)")
    print(f"✓ Saved to: 03_burst1_data_114.txt through 03_burst4_data_114.txt")
    print()
    
    # Build FCCH burst
    print("Step 4: Building FCCH Burst")
    print("-" * 80)
    fcch = build_fcch_burst()
    print(f"FCCH burst length: {len(fcch)} bits")
    print(f"FCCH burst (first 32): {fcch[:32]} (all zeros)")
    save_bit_sequence("04_fcch_burst.txt", fcch, "FCCH Burst (148 bits, all zeros)")
    print("✓ Saved to: 04_fcch_burst.txt")
    print()
    
    # Build SCH burst
    print("Step 5: Building SCH Burst")
    print("-" * 80)
    bsic = 10
    fn = 0
    sch = build_sch_burst(bsic, fn)
    print(f"SCH burst length: {len(sch)} bits")
    print(f"BSIC: {bsic}, Frame Number: {fn}")
    print(f"SCH burst (first 32): {sch[:32]}")
    save_bit_sequence("05_sch_burst.txt", sch, 
                     f"SCH Burst (148 bits, BSIC={bsic}, FN={fn})")
    print("✓ Saved to: 05_sch_burst.txt")
    print()
    
    # Build normal bursts
    print("Step 6: Building Normal Bursts")
    print("-" * 80)
    tsc = get_tsc(0)
    print(f"Using TSC 0: {tsc}")
    normal_bursts = []
    for i, data114 in enumerate(data_bursts):
        burst = build_normal_burst(data114, tsc)
        normal_bursts.append(burst)
        print(f"  Normal burst {i+1} length: {len(burst)} bits")
        print(f"  Normal burst {i+1} (first 32): {burst[:32]}")
        save_bit_sequence(f"06_normal_burst{i+1}_148.txt", burst,
                         f"Normal Burst {i+1} (148 bits with TSC 0)")
    print(f"✓ Saved to: 06_normal_burst1_148.txt through 06_normal_burst4_148.txt")
    print()
    
    # Assemble complete frame
    frame_bursts = [fcch, sch] + normal_bursts
    print("Step 7: Frame Structure")
    print("-" * 80)
    print(f"Complete frame: {len(frame_bursts)} bursts")
    print(f"  Burst 1: FCCH (frequency correction)")
    print(f"  Burst 2: SCH (synchronization)")
    print(f"  Bursts 3-6: Normal bursts (BCCH data)")
    
    # Save concatenated burst sequence
    all_burst_bits = np.concatenate(frame_bursts)
    save_bit_sequence("07_all_bursts_concatenated.txt", all_burst_bits,
                     f"All Bursts Concatenated ({len(all_burst_bits)} bits)")
    print(f"✓ Saved to: 07_all_bursts_concatenated.txt")
    print()
    
    # Modulate bursts to IQ samples
    print("Step 8: GMSK Modulation to IQ Samples")
    print("-" * 80)
    guard_samples = 66  # ~8.25 bit periods * 8 samples/bit
    iq_samples = modulate_burst_sequence(
        frame_bursts,
        guard_samples=guard_samples
    )
    
    print(f"Generated {len(iq_samples)} IQ samples")
    print(f"Sample rate: {FS_GEN:.2f} Hz ({FS_GEN/1e6:.3f} MHz)")
    duration_ms = len(iq_samples) / FS_GEN * 1000
    print(f"Duration: {duration_ms:.2f} ms")
    print(f"IQ samples (first 8 real parts): {np.real(iq_samples[:8])}")
    print(f"IQ samples (first 8 imag parts): {np.imag(iq_samples[:8])}")
    print()
    
    # Save to cfile
    output_file = "bcch_waveform_debug.cfile"
    print(f"Saving waveform to '{output_file}'...")
    write_complex_iq(output_file, iq_samples)
    print("✓ Saved to: bcch_waveform_debug.cfile")
    print()
    
    # Save IQ data as text for debugging
    print("Saving IQ samples to text file...")
    with open("08_iq_samples.txt", 'w') as f:
        f.write(f"# IQ Samples (Complex)\n")
        f.write(f"# Total samples: {len(iq_samples)}\n")
        f.write(f"# Format: Real Imaginary\n")
        f.write("#" + "=" * 70 + "\n\n")
        for i, sample in enumerate(iq_samples[:1000]):  # Save first 1000 samples
            f.write(f"{np.real(sample):.6f} {np.imag(sample):.6f}\n")
        if len(iq_samples) > 1000:
            f.write(f"\n... ({len(iq_samples) - 1000} more samples not shown)\n")
    print("✓ Saved to: 08_iq_samples.txt (first 1000 samples)")
    print()
    
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print()
    print("Files created:")
    print("  01_info_bits.txt               - Original information (184 bits)")
    print("  02_encoded_456.txt             - After FIRE + convolutional (456 bits)")
    print("  03_burst[1-4]_data_114.txt     - Interleaved data payloads (114 bits each)")
    print("  04_fcch_burst.txt              - FCCH burst (148 bits)")
    print("  05_sch_burst.txt               - SCH burst (148 bits)")
    print("  06_normal_burst[1-4]_148.txt   - Normal bursts (148 bits each)")
    print("  07_all_bursts_concatenated.txt - All bursts together")
    print("  08_iq_samples.txt              - IQ samples (first 1000)")
    print("  bcch_waveform_debug.cfile      - Complete waveform for SDR")
    print()
    print("To transmit with HackRF One:")
    print(f"  hackrf_transfer -t bcch_waveform_debug.cfile -f 935000000 -s {int(FS_GEN)} -a 1 -x 20")
    print()
    print("WARNING: Transmission on GSM frequencies may be illegal!")
    print("Always use proper attenuation and check local regulations.")
    print()


if __name__ == "__main__":
    main()
