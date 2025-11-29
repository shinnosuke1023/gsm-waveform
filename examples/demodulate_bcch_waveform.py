#!/usr/bin/env python3
"""Example: Demodulate GSM BCCH Waveform

This script demonstrates how to read a BCCH waveform file,
demodulate it, and decode the information.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

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
    OSR_DEFAULT,
    format_system_information
)


def format_bcch_content(info_bits: np.ndarray) -> str:
    """Format BCCH information bits for display.
    
    BCCH carries System Information messages. This function provides
    a basic interpretation of the 184-bit information block.
    
    Args:
        info_bits: 184-bit information array
        
    Returns:
        Formatted string with BCCH content breakdown
    """
    output = []
    output.append("")
    output.append("=" * 60)
    output.append("BCCH Information Content")
    output.append("=" * 60)
    output.append("")
    
    # Total bits
    output.append(f"Total information bits: {len(info_bits)}")
    output.append("")
    
    # Show bit representation in groups
    output.append("Bit content (grouped by 8 bits):")
    output.append("-" * 60)
    
    for i in range(0, min(len(info_bits), 184), 8):
        byte_bits = info_bits[i:i+8]
        # Convert to integer value
        byte_val = int(''.join(str(b) for b in byte_bits), 2) if len(byte_bits) == 8 else 0
        
        # Format as: [offset] bits -> hex (decimal)
        bit_str = ''.join(str(b) for b in byte_bits)
        output.append(f"  [{i:3d}-{i+7:3d}] {bit_str} -> 0x{byte_val:02X} ({byte_val:3d})")
    
    output.append("")
    
    # Statistics
    ones_count = np.sum(info_bits)
    zeros_count = len(info_bits) - ones_count
    output.append("Bit statistics:")
    output.append(f"  Ones:  {ones_count:3d} ({ones_count/len(info_bits)*100:.1f}%)")
    output.append(f"  Zeros: {zeros_count:3d} ({zeros_count/len(info_bits)*100:.1f}%)")
    output.append("")
    
    # Note about System Information types
    output.append("Note: BCCH carries GSM System Information messages.")
    output.append("      Common message types include:")
    output.append("      - System Information Type 1: BCCH allocation")
    output.append("      - System Information Type 2: Neighbor cell description")
    output.append("      - System Information Type 3: Cell identity and location")
    output.append("      - System Information Type 4: CBCH and location area info")
    output.append("")
    output.append("      Full decoding requires L3 message parsing (see GSM 04.08)")
    output.append("=" * 60)
    output.append("")
    
    return '\n'.join(output)


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
    
    # 2. Demodulate to bits (try all phase alignments)
    print("Demodulating IQ samples to bits...")
    all_phases = gmsk_demodulate(iq_samples, osr=OSR_DEFAULT, return_all_phases=True)
    print(f"Got {len(all_phases)} phase alignments")
    print()
    
    # 3. Try decoding with each phase alignment
    print("Trying to decode BCCH from each phase alignment...")
    best_valid = False
    best_info_bits = None
    best_phase = 0
    
    for phase_idx, demod_bits in enumerate(all_phases):
        # Strategy 1: Try known frame structure (FCCH + SCH + 4 normal bursts)
        # With guard_samples=0, bursts are back-to-back at positions:
        # FCCH: 0-147, SCH: 148-295, Normal bursts: 296+
        for offset in range(-10, 11):
            data_bursts = []
            success = True
            
            for i in range(4):
                # Normal bursts start at position 296 (after FCCH+SCH)
                burst_start = 296 + offset + i * 148
                
                if burst_start >= 0 and burst_start + 148 <= len(demod_bits):
                    burst = demod_bits[burst_start:burst_start + 148]
                    try:
                        data114 = extract_burst_data_114(burst)
                        data_bursts.append(data114)
                    except:
                        success = False
                        break
                else:
                    success = False
                    break
            
            if success and len(data_bursts) == 4:
                try:
                    info_bits, valid = decode_bcch_pipeline(data_bursts)
                    
                    if valid:
                        print(f"  Phase {phase_idx}: BCCH decoded successfully (known structure, offset={offset:+d})!")
                        best_valid = True
                        best_info_bits = info_bits
                        best_phase = phase_idx
                        break
                except Exception:
                    pass
        
        if best_valid:
            break
        
        # Strategy 2: Try using TSC correlation to find normal bursts
        if not best_valid:
            burst_positions = detect_burst_by_tsc(demod_bits, tsc_index=tsc_index, threshold=0.5)
            
            if len(burst_positions) >= 4:
                # Try different groups of 4 consecutive detected bursts
                for start_idx in range(min(5, len(burst_positions) - 3)):
                    for offset in range(-5, 6):
                        data_bursts = []
                        success = True
                        
                        for i in range(4):
                            if start_idx + i >= len(burst_positions):
                                success = False
                                break
                            
                            pos = burst_positions[start_idx + i]
                            burst_start = pos - 61 + offset
                            
                            if burst_start >= 0 and burst_start + 148 <= len(demod_bits):
                                burst = demod_bits[burst_start:burst_start + 148]
                                try:
                                    data114 = extract_burst_data_114(burst)
                                    data_bursts.append(data114)
                                except:
                                    success = False
                                    break
                            else:
                                success = False
                                break
                        
                        if success and len(data_bursts) == 4:
                            try:
                                info_bits, valid = decode_bcch_pipeline(data_bursts)
                                
                                if valid:
                                    print(f"  Phase {phase_idx}: BCCH decoded successfully (TSC correlation, start_idx={start_idx}, offset={offset:+d})!")
                                    best_valid = True
                                    best_info_bits = info_bits
                                    best_phase = phase_idx
                                    break
                            except Exception:
                                pass
                    
                    if best_valid:
                        break
        
        if best_valid:
            break
        
        # If direct decoding failed, try SCH-based detection
        sch_positions = detect_sch_burst(demod_bits, threshold=0.5)
        
        if sch_positions:
            # Try to decode SCH
            sch_pos = sch_positions[0]
            if sch_pos >= 42:
                sch_burst = demod_bits[sch_pos-42:sch_pos+106]
                if len(sch_burst) >= 148:
                    left_data, right_data = extract_sch_data(sch_burst)
                    sch39 = left_data
                    bsic, fn, valid = decode_sch_39bits(sch39)
                    
                    if valid and not best_valid:
                        print(f"  Phase {phase_idx}: SCH decoded (BSIC={bsic}, FN={fn})")
        
        # Try multiple thresholds for burst detection
        for threshold in [0.6, 0.55, 0.5, 0.45]:
            burst_positions = detect_burst_by_tsc(demod_bits, tsc_index=tsc_index, threshold=threshold)
            
            if len(burst_positions) >= 4:
                # Try different starting positions (skip FCCH/SCH if detected)
                for start_idx in range(min(3, len(burst_positions) - 3)):
                    # Try small offsets around detected position
                    for offset in range(-5, 6):
                        data_bursts = []
                        success = True
                        
                        for i in range(4):
                            if start_idx + i >= len(burst_positions):
                                success = False
                                break
                            
                            pos = burst_positions[start_idx + i]
                            burst_start = pos - 61 + offset
                            
                            if burst_start >= 0 and burst_start + 148 <= len(demod_bits):
                                burst = demod_bits[burst_start:burst_start + 148]
                                data114 = extract_burst_data_114(burst)
                                data_bursts.append(data114)
                            else:
                                success = False
                                break
                        
                        if success and len(data_bursts) == 4:
                            # Try to decode BCCH
                            try:
                                info_bits, valid = decode_bcch_pipeline(data_bursts)
                                
                                if valid and not best_valid:
                                    print(f"  Phase {phase_idx}: BCCH decoded successfully (threshold={threshold}, start_idx={start_idx}, offset={offset:+d})!")
                                    best_valid = True
                                    best_info_bits = info_bits
                                    best_phase = phase_idx
                                    break
                            except Exception:
                                pass
                    
                    if best_valid:
                        break
                
                if best_valid:
                    break
            
            if best_valid:
                break
        
        # If TSC detection failed, try timing-based approach using SCH position
        # This assumes guard_samples=0 in waveform generation
        if not best_valid and sch_positions:
            sch_pos = sch_positions[0]
            # SCH TSC center is at sch_pos, so SCH starts at sch_pos - 61
            sch_start = sch_pos - 61
            
            # With guard_samples=0, bursts are back-to-back
            # First normal burst starts immediately after SCH
            first_normal_start = sch_start + 148
            
            # Try small offsets to account for demodulation alignment
            for offset in range(-5, 6):
                data_bursts = []
                success = True
                
                for i in range(4):
                    burst_start = first_normal_start + offset + i * 148
                    
                    if burst_start >= 0 and burst_start + 148 <= len(demod_bits):
                        burst = demod_bits[burst_start:burst_start + 148]
                        try:
                            data114 = extract_burst_data_114(burst)
                            data_bursts.append(data114)
                        except:
                            success = False
                            break
                    else:
                        success = False
                        break
                
                if success and len(data_bursts) == 4:
                    try:
                        info_bits, valid = decode_bcch_pipeline(data_bursts)
                        
                        if valid:
                            print(f"  Phase {phase_idx}: BCCH decoded successfully (timing-based, offset={offset:+d})!")
                            best_valid = True
                            best_info_bits = info_bits
                            best_phase = phase_idx
                            break
                    except Exception:
                        pass
        
        if best_valid:
            break
    
    print()
    
    # 4. Display results
    if best_valid:
        print("=" * 60)
        print(f"Successfully decoded using phase {best_phase}")
        print("=" * 60)
        print()
        print(f"  ✓ BCCH decoded successfully!")
        print(f"  Parity check: PASSED")
        
        # Display System Information if present
        formatted_si = format_system_information(best_info_bits)
        print(formatted_si)
        
        # Also display raw BCCH content for debugging
        formatted_content = format_bcch_content(best_info_bits)
        print(formatted_content)
    else:
        print(f"  ✗ BCCH decoding failed in all phase alignments")
        print(f"  This may be due to:")
        print(f"    - Noise in the signal")
        print(f"    - Incorrect burst alignment")
        print(f"    - Wrong TSC index")
        print(f"    - Wrong file format or corrupted data")
    
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
