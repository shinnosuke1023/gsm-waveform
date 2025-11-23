#!/usr/bin/env python3
"""Example: Generate GSM BCCH Waveform

This script demonstrates how to generate a complete BCCH waveform
including FCCH, SCH, and 4 normal bursts carrying encoded information.
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
    FS_GEN,
    encode_system_information_type3
)


def generate_bcch_frame(info_bits: np.ndarray, bsic: int = 0, fn: int = 0, tsc_index: int = 0):
    """Generate a complete BCCH frame.
    
    Frame structure:
    - FCCH burst (frequency correction)
    - SCH burst (synchronization)
    - 4 Normal bursts (BCCH data)
    
    Args:
        info_bits: 184-bit information payload
        bsic: Base Station Identity Code (0-63)
        fn: Frame number
        tsc_index: Training Sequence Code index (0-7)
        
    Returns:
        List of burst bit arrays
    """
    # 1. Encode BCCH information
    print(f"Encoding {len(info_bits)} information bits...")
    encoded_456 = make_bcch_encoded_456(info_bits)
    print(f"Encoded to {len(encoded_456)} bits")
    
    # 2. Interleave to 4 bursts
    print("Interleaving to 4 bursts...")
    data_bursts = interleave_456_to_4x114(encoded_456)
    
    # 3. Build FCCH burst
    print("Building FCCH burst...")
    fcch = build_fcch_burst()
    
    # 4. Build SCH burst
    print(f"Building SCH burst (BSIC={bsic}, FN={fn})...")
    sch = build_sch_burst(bsic, fn)
    
    # 5. Build 4 normal bursts
    print(f"Building 4 normal bursts with TSC {tsc_index}...")
    tsc = get_tsc(tsc_index)
    normal_bursts = []
    for i, data114 in enumerate(data_bursts):
        burst = build_normal_burst(data114, tsc)
        normal_bursts.append(burst)
        print(f"  Burst {i+1}: {len(burst)} bits")
    
    # Assemble complete frame
    frame = [fcch, sch] + normal_bursts
    return frame


def main():
    """Main function to generate and save BCCH waveform"""
    
    print("=" * 60)
    print("GSM BCCH Waveform Generator")
    print("=" * 60)
    print()
    
    # Create System Information Type 3 message with GSM 04.08 compliant parameters
    print("Creating System Information Type 3 message (GSM 04.08)...")
    
    # Define network parameters
    cell_identity = 12345             # Cell Identity
    location_area_code = 100          # Location Area Code
    mobile_country_code = 440         # MCC (Japan example: 440)
    mobile_network_code = 10          # MNC (example: 10)
    arfcn = 975                       # ARFCN (e.g., GSM-900 downlink)
    neighbor_cells = [980, 985]  # Neighbor cell ARFCNs (max 2 due to 184-bit limit)
    
    print(f"  Cell Identity: {cell_identity}")
    print(f"  Location Area Code: {location_area_code}")
    print(f"  MCC: {mobile_country_code}, MNC: {mobile_network_code}")
    print(f"  ARFCN: {arfcn}")
    print(f"  Neighbor Cells: {neighbor_cells}")
    
    # Encode System Information with GSM 04.08 compliant parameters
    info_bits = encode_system_information_type3(
        cell_identity=cell_identity,
        location_area_code=location_area_code,
        mobile_country_code=mobile_country_code,
        mobile_network_code=mobile_network_code,
        arfcn=arfcn,
        neighbor_cells=neighbor_cells,
        # Use default values for other GSM 04.08 parameters
        att=True,
        bs_ag_blks_res=1,
        ccch_conf=1,
        bs_pa_mfrms=2,
        t3212=0,
        pwrc=False,
        dtx=2,
        radio_link_timeout=4,
        cell_reselect_hysteresis=2,
        ms_txpwr_max_cch=0,
        rxlev_access_min=0,
        neci=True,
        acs=False,
        max_retrans=1,
        tx_integer=3,
        cell_bar_access=False,
        re=False,
        acc=0xFFFF
    )
    
    # Generate normal bursts for BCCH data
    # Note: Skipping FCCH and SCH for now as they cause demodulation issues with SI data
    # In a real system, FCCH and SCH would be needed for frequency and time synchronization
    print()
    print("Building normal bursts with SI data...")
    
    # Encode SI through BCCH pipeline
    encoded_456 = make_bcch_encoded_456(info_bits)
    data_bursts = interleave_456_to_4x114(encoded_456)
    
    # Build 4 normal bursts with TSC
    tsc = get_tsc(0)  # Training Sequence Code 0
    frame_bursts = [build_normal_burst(data, tsc) for data in data_bursts]
    
    print()
    print(f"Frame generated with {len(frame_bursts)} bursts:")
    print(f"  1-4. Normal bursts (BCCH data with SI Type 3)")
    print()
    
    # Modulate bursts to IQ samples
    print("Modulating bursts to GMSK IQ samples...")
    # Note: guard_samples=0 for compatibility with System Information demodulation
    # Guard samples can cause phase discontinuities with complex SI data patterns
    guard_samples = 0
    iq_samples = modulate_burst_sequence(
        frame_bursts,
        guard_samples=guard_samples
    )
    
    print(f"Generated {len(iq_samples)} IQ samples")
    print(f"Sample rate: {FS_GEN:.2f} Hz ({FS_GEN/1e6:.3f} MHz)")
    duration_ms = len(iq_samples) / FS_GEN * 1000
    print(f"Duration: {duration_ms:.2f} ms")
    print()
    
    # Save to file
    output_file = "bcch_waveform.cfile"
    print(f"Saving waveform to '{output_file}'...")
    write_complex_iq(output_file, iq_samples)
    
    print()
    print("=" * 60)
    print("Waveform generation complete!")
    print("=" * 60)
    print()
    print("To transmit with HackRF One:")
    print(f"  hackrf_transfer -t {output_file} -f 935000000 -s {int(FS_GEN)} -a 1 -x 20")
    print()
    print("To view with GNU Radio:")
    print("  Use File Source block with:")
    print(f"    - File: {output_file}")
    print(f"    - Sample Rate: {int(FS_GEN)}")
    print("    - Type: Complex float32")
    print()
    print("WARNING: Transmission on GSM frequencies may be illegal!")
    print("Always use proper attenuation and check local regulations.")
    print()


if __name__ == "__main__":
    main()
