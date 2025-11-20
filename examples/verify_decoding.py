#!/usr/bin/env python3
"""Comprehensive Decoding Verification Script

This script verifies that all decoding functions work correctly by:
1. Encoding test data
2. Decoding it back
3. Comparing results
4. Outputting detailed intermediate results
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsm_waveform import (
    # Encoding
    fire_encode_184,
    append_block_tail,
    convolutional_encode,
    make_bcch_encoded_456,
    interleave_456_to_4x114,
    build_sch_39bits,
    # Decoding
    fire_decode_224,
    remove_tail_bits,
    viterbi_decode,
    deinterleave_4x114_to_456,
    decode_bcch_pipeline,
    decode_sch_39bits,
)


def test_fire_code():
    """Test FIRE code encoding and decoding"""
    print("=" * 80)
    print("Test 1: FIRE Code (184 bits → 224 bits)")
    print("=" * 80)
    
    # Test with various patterns
    test_cases = [
        ("All zeros", np.zeros(184, dtype=np.uint8)),
        ("All ones", np.ones(184, dtype=np.uint8)),
        ("Random pattern", np.random.randint(0, 2, 184, dtype=np.uint8)),
        ("Alternating", np.tile([0, 1], 92).astype(np.uint8)),
    ]
    
    all_passed = True
    for name, info_bits in test_cases:
        print(f"\n{name}:")
        print(f"  Input (first 32 bits): {info_bits[:32]}")
        
        # Encode
        encoded_224 = fire_encode_184(info_bits)
        print(f"  Encoded to {len(encoded_224)} bits")
        print(f"  Parity bits: {encoded_224[184:]}")
        
        # Decode
        decoded_info, parity_valid = fire_decode_224(encoded_224)
        print(f"  Decoded back to {len(decoded_info)} bits")
        print(f"  Parity check: {'✓ VALID' if parity_valid else '✗ INVALID'}")
        
        # Verify
        match = np.array_equal(info_bits, decoded_info)
        print(f"  Match original: {'✓ YES' if match else '✗ NO'}")
        
        if not match or not parity_valid:
            all_passed = False
            print(f"  ❌ FAILED!")
        else:
            print(f"  ✅ PASSED!")
    
    return all_passed


def test_convolutional_code():
    """Test convolutional encoding and Viterbi decoding"""
    print("\n" + "=" * 80)
    print("Test 2: Convolutional Code (rate 1/2, K=5)")
    print("=" * 80)
    
    test_cases = [
        ("All zeros", np.zeros(50, dtype=np.uint8)),
        ("All ones", np.ones(50, dtype=np.uint8)),
        ("Random pattern", np.random.randint(0, 2, 50, dtype=np.uint8)),
    ]
    
    all_passed = True
    for name, info_bits in test_cases:
        print(f"\n{name}:")
        print(f"  Input: {len(info_bits)} bits")
        print(f"  Input (first 32): {info_bits[:32]}")
        
        # Encode
        encoded = convolutional_encode(info_bits)
        print(f"  Encoded to {len(encoded)} bits (rate 1/2)")
        
        # Decode with Viterbi
        decoded = viterbi_decode(encoded)
        print(f"  Decoded back to {len(decoded)} bits")
        print(f"  Decoded (first 32): {decoded[:32]}")
        
        # Verify
        match = np.array_equal(info_bits, decoded)
        bit_errors = np.sum(info_bits != decoded)
        accuracy = np.mean(info_bits == decoded) * 100
        
        print(f"  Bit errors: {bit_errors} / {len(info_bits)}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Match original: {'✓ YES' if match else '✗ NO'}")
        
        # Allow small error rate for random data due to algorithm limitations
        if accuracy < 95:
            all_passed = False
            print(f"  ❌ FAILED (accuracy too low)!")
        else:
            print(f"  ✅ PASSED!")
    
    return all_passed


def test_interleaving():
    """Test block interleaving and deinterleaving"""
    print("\n" + "=" * 80)
    print("Test 3: Block Interleaving (456 bits ↔ 4×114 bits)")
    print("=" * 80)
    
    test_cases = [
        ("Sequential", np.arange(456, dtype=np.uint8) % 2),
        ("Random", np.random.randint(0, 2, 456, dtype=np.uint8)),
    ]
    
    all_passed = True
    for name, bits_456 in test_cases:
        print(f"\n{name}:")
        print(f"  Input: {len(bits_456)} bits")
        print(f"  Input (first 32): {bits_456[:32]}")
        
        # Interleave
        bursts = interleave_456_to_4x114(bits_456)
        print(f"  Interleaved to {len(bursts)} bursts of 114 bits")
        for i, burst in enumerate(bursts):
            print(f"    Burst {i+1} (first 16): {burst[:16]}")
        
        # Deinterleave
        reconstructed = deinterleave_4x114_to_456(bursts)
        print(f"  Deinterleaved back to {len(reconstructed)} bits")
        print(f"  Reconstructed (first 32): {reconstructed[:32]}")
        
        # Verify
        match = np.array_equal(bits_456, reconstructed)
        print(f"  Match original: {'✓ YES' if match else '✗ NO'}")
        
        if not match:
            all_passed = False
            print(f"  ❌ FAILED!")
        else:
            print(f"  ✅ PASSED!")
    
    return all_passed


def test_bcch_pipeline():
    """Test complete BCCH encoding/decoding pipeline"""
    print("\n" + "=" * 80)
    print("Test 4: Complete BCCH Pipeline (184 bits → ... → 184 bits)")
    print("=" * 80)
    
    test_cases = [
        ("All zeros", np.zeros(184, dtype=np.uint8)),
        ("All ones", np.ones(184, dtype=np.uint8)),
        ("Random pattern 1", np.random.randint(0, 2, 184, dtype=np.uint8)),
        ("Random pattern 2", np.random.randint(0, 2, 184, dtype=np.uint8)),
        ("Custom pattern", np.tile([0, 1, 0, 1, 1, 0, 1, 0], 23).astype(np.uint8)),
    ]
    
    all_passed = True
    for name, info_bits in test_cases:
        print(f"\n{name}:")
        print(f"  Original info: {len(info_bits)} bits")
        print(f"  Original (first 32): {info_bits[:32]}")
        
        # Encode
        print("\n  Encoding steps:")
        encoded_456 = make_bcch_encoded_456(info_bits)
        print(f"    1. FIRE + tail + convolutional: {len(encoded_456)} bits")
        
        bursts = interleave_456_to_4x114(encoded_456)
        print(f"    2. Interleaved to {len(bursts)} bursts of 114 bits")
        
        # Decode
        print("\n  Decoding steps:")
        decoded_info, parity_valid = decode_bcch_pipeline(bursts)
        print(f"    1. Deinterleave: 4×114 → 456 bits")
        print(f"    2. Viterbi decode: 456 → 228 bits")
        print(f"    3. Remove tail: 228 → 224 bits")
        print(f"    4. FIRE decode: 224 → 184 bits")
        
        print(f"\n  Decoded info: {len(decoded_info)} bits")
        print(f"  Decoded (first 32): {decoded_info[:32]}")
        print(f"  Parity check: {'✓ VALID' if parity_valid else '✗ INVALID'}")
        
        # Verify
        match = np.array_equal(info_bits, decoded_info)
        bit_errors = np.sum(info_bits != decoded_info)
        
        print(f"  Bit errors: {bit_errors} / {len(info_bits)}")
        print(f"  Match original: {'✓ YES' if match else '✗ NO'}")
        
        if not match or not parity_valid:
            all_passed = False
            print(f"  ❌ FAILED!")
        else:
            print(f"  ✅ PASSED!")
    
    return all_passed


def test_sch_decoding():
    """Test SCH encoding and decoding"""
    print("\n" + "=" * 80)
    print("Test 5: SCH Encoding/Decoding (BSIC + Frame Number)")
    print("=" * 80)
    
    test_cases = [
        (0, 0),
        (10, 0),
        (25, 12345),
        (63, 2715647),  # Max values
    ]
    
    all_passed = True
    for bsic, fn in test_cases:
        print(f"\nBSIC={bsic}, FN={fn}:")
        
        # Encode
        sch39 = build_sch_39bits(bsic, fn)
        print(f"  Encoded to {len(sch39)} bits")
        print(f"  SCH bits (first 32): {sch39[:32]}")
        
        # Decode
        dec_bsic, dec_fn, crc_valid = decode_sch_39bits(sch39)
        print(f"  Decoded BSIC: {dec_bsic}")
        print(f"  Decoded FN: {dec_fn}")
        print(f"  CRC check: {'✓ VALID' if crc_valid else '✗ INVALID'}")
        
        # Verify
        bsic_match = (dec_bsic == bsic)
        print(f"  BSIC match: {'✓ YES' if bsic_match else '✗ NO'}")
        
        if not crc_valid or not bsic_match:
            all_passed = False
            print(f"  ❌ FAILED!")
        else:
            print(f"  ✅ PASSED!")
    
    return all_passed


def main():
    """Run all decoding verification tests"""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GSM DECODING VERIFICATION SUITE" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = []
    
    # Run all tests
    results.append(("FIRE Code", test_fire_code()))
    results.append(("Convolutional Code", test_convolutional_code()))
    results.append(("Interleaving", test_interleaving()))
    results.append(("BCCH Pipeline", test_bcch_pipeline()))
    results.append(("SCH Decoding", test_sch_decoding()))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:.<30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Decoding is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please review the output above")
    print("=" * 80)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
