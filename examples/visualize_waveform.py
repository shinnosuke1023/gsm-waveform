#!/usr/bin/env python3
"""Visualize GSM Waveform Characteristics

This script demonstrates basic waveform analysis without requiring matplotlib.
Shows waveform statistics and characteristics in text format.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import read_complex_iq, get_sample_info, FS_GEN


def analyze_waveform(filename: str):
    """Analyze and display waveform characteristics"""
    
    print("=" * 60)
    print("GSM Waveform Analysis")
    print("=" * 60)
    print()
    
    # File info
    info = get_sample_info(filename)
    print("File Information:")
    print(f"  File: {info['filename']}")
    print(f"  Size: {info['file_size_bytes']:,} bytes ({info['file_size_bytes']/1024:.1f} KB)")
    print(f"  Samples: {info['num_samples']:,}")
    print(f"  Data type: {info['dtype']}")
    print()
    
    # Read samples
    iq = read_complex_iq(filename)
    
    # Time domain analysis
    print("Time Domain Statistics:")
    duration_ms = len(iq) / FS_GEN * 1000
    print(f"  Duration: {duration_ms:.3f} ms")
    print(f"  Sample rate: {FS_GEN:.2f} Hz ({FS_GEN/1e6:.3f} MHz)")
    print()
    
    # Magnitude analysis
    magnitude = np.abs(iq)
    print("Magnitude Statistics:")
    print(f"  Min: {np.min(magnitude):.6f}")
    print(f"  Max: {np.max(magnitude):.6f}")
    print(f"  Mean: {np.mean(magnitude):.6f}")
    print(f"  Std: {np.std(magnitude):.6f}")
    print()
    
    # Phase analysis
    phase = np.angle(iq)
    print("Phase Statistics:")
    print(f"  Range: [{np.min(phase):.3f}, {np.max(phase):.3f}] radians")
    print(f"  Mean: {np.mean(phase):.3f} radians")
    print()
    
    # Power analysis
    power = magnitude ** 2
    print("Power Statistics:")
    print(f"  Mean power: {np.mean(power):.6f}")
    print(f"  Peak power: {np.max(power):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(power)):.6f}")
    print()
    
    # Phase continuity (check for discontinuities)
    phase_diff = np.diff(phase)
    # Unwrap to handle phase wrapping
    phase_diff_unwrapped = np.diff(np.unwrap(phase))
    print("Phase Continuity:")
    print(f"  Max phase jump: {np.max(np.abs(phase_diff)):.3f} radians")
    print(f"  Mean phase change: {np.mean(np.abs(phase_diff_unwrapped)):.6f} radians/sample")
    print()
    
    # Data quality checks
    print("Data Quality:")
    has_nan = np.any(np.isnan(iq))
    has_inf = np.any(np.isinf(iq))
    has_zero = np.any(magnitude == 0)
    print(f"  NaN values: {'Yes' if has_nan else 'No'}")
    print(f"  Inf values: {'Yes' if has_inf else 'No'}")
    print(f"  Zero magnitude samples: {'Yes' if has_zero else 'No'}")
    
    if has_zero:
        zero_count = np.sum(magnitude == 0)
        print(f"    (Count: {zero_count}, {zero_count/len(iq)*100:.2f}%)")
    
    print()
    
    # Simple histogram of magnitudes (text-based)
    print("Magnitude Distribution (10 bins):")
    hist, bin_edges = np.histogram(magnitude, bins=10)
    max_bar_width = 50
    max_count = np.max(hist)
    
    for i, count in enumerate(hist):
        bar_width = int(count / max_count * max_bar_width) if max_count > 0 else 0
        bar = '█' * bar_width
        print(f"  {bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}: {bar} {count}")
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GSM waveform IQ file')
    parser.add_argument('filename', nargs='?', default='bcch_waveform.cfile',
                       help='IQ file to analyze (default: bcch_waveform.cfile)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found")
        print()
        print("Generate a waveform first with:")
        print("  python examples/generate_bcch_waveform.py")
        sys.exit(1)
    
    analyze_waveform(args.filename)


if __name__ == "__main__":
    main()
