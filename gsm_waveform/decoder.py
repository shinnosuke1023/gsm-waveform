"""GSM Channel Decoding

Implements Viterbi decoding and FIRE code verification
for GSM broadcast channels.
"""

import numpy as np
from typing import Tuple, Optional
from .constants import CONV_K, CONV_GENERATORS


def deinterleave_4x114_to_456(bursts: list) -> np.ndarray:
    """Reverse block interleaving: [114, 114, 114, 114] → 456 bits.
    
    This is the inverse of interleave_456_to_4x114.
    Bits are read row-wise and written column-wise.
    
    Args:
        bursts: List of 4 numpy arrays, each containing 114 bits
        
    Returns:
        456-bit array
    """
    assert len(bursts) == 4, f"Expected 4 bursts, got {len(bursts)}"
    for i, burst in enumerate(bursts):
        assert len(burst) == 114, f"Burst {i} has {len(burst)} bits, expected 114"
    
    M = 4    # Number of bursts
    
    # Create matrix from bursts (each burst is a row)
    mat = np.array(bursts, dtype=np.uint8)
    
    # Read column-wise to reconstruct original sequence
    bits456 = np.zeros(456, dtype=np.uint8)
    for k in range(456):
        r = k % M
        c = k // M
        bits456[k] = mat[r, c]
    
    return bits456


def viterbi_decode(encoded_bits: np.ndarray,
                  generators: Tuple[int, int] = CONV_GENERATORS,
                  K: int = CONV_K) -> np.ndarray:
    """Viterbi decoder for convolutional code (rate 1/2, constraint length K).
    
    Args:
        encoded_bits: Received encoded bit sequence (length must be even)
        generators: Tuple of generator polynomials in octal
        K: Constraint length
        
    Returns:
        Decoded bit sequence (length = len(encoded_bits) / 2)
    """
    assert len(encoded_bits) % 2 == 0, "Encoded bits length must be even"
    
    n_bits = len(encoded_bits) // 2  # Number of information bits
    n_states = 2 ** (K - 1)  # Number of states
    
    # Initialize metrics and paths
    # Use log-likelihood for numerical stability
    path_metrics = np.full(n_states, np.inf)
    path_metrics[0] = 0  # Start from all-zero state
    
    # Store survivor paths
    survivors = np.zeros((n_bits, n_states), dtype=np.int32)
    survivor_states = np.zeros((n_bits, n_states), dtype=np.int32)
    
    # Process each bit
    for i in range(n_bits):
        # Get received symbol pair
        r0 = encoded_bits[2 * i]
        r1 = encoded_bits[2 * i + 1]
        received = np.array([r0, r1], dtype=np.uint8)
        
        # New metrics for this iteration
        new_metrics = np.full(n_states, np.inf)
        
        # For each current state
        for state in range(n_states):
            if path_metrics[state] == np.inf:
                continue  # Skip unreachable states
            
            # Try both input bits (0 and 1)
            for input_bit in [0, 1]:
                # Compute expected output for this state and input
                expected = _compute_conv_output(state, input_bit, generators, K)
                
                # Compute Hamming distance (branch metric)
                distance = np.sum(expected != received)
                
                # Compute new path metric
                metric = path_metrics[state] + distance
                
                # Compute next state
                next_state = _get_next_state(state, input_bit, K)
                
                # Update if this path is better
                if metric < new_metrics[next_state]:
                    new_metrics[next_state] = metric
                    survivors[i, next_state] = input_bit
                    survivor_states[i, next_state] = state
        
        # Update metrics
        path_metrics = new_metrics
    
    # Traceback from best final state
    # Prefer state 0 (all tails should drive to zero state)
    if path_metrics[0] != np.inf:
        final_state = 0
    else:
        final_state = np.argmin(path_metrics)
    
    # Decode bits by tracing back
    decoded = np.zeros(n_bits, dtype=np.uint8)
    current_state = final_state
    
    for i in range(n_bits - 1, -1, -1):
        decoded[i] = survivors[i, current_state]
        current_state = survivor_states[i, current_state]
    
    return decoded


def _compute_conv_output(state: int, input_bit: int,
                        generators: Tuple[int, int],
                        K: int) -> np.ndarray:
    """Compute convolutional encoder output for given state and input.
    
    Args:
        state: Current encoder state (shift register contents)
        input_bit: Input bit (0 or 1)
        generators: Generator polynomials
        K: Constraint length
        
    Returns:
        Output bit pair [g0_out, g1_out]
    """
    outputs = []
    for g in generators:
        y = 0
        # Generator bit i corresponds to tap for D^i
        # i=0 is current input, i=1..K-1 are state bits
        for i in range(K):
            if (g >> i) & 1:
                if i == 0:
                    y ^= input_bit
                else:
                    # State bit: LSB of state is most recent bit (i=1)
                    y ^= (state >> (i - 1)) & 1
        outputs.append(y)
    
    return np.array(outputs, dtype=np.uint8)


def _get_next_state(state: int, input_bit: int, K: int) -> int:
    """Compute next state after shifting in input bit.
    
    Args:
        state: Current state
        input_bit: Input bit
        K: Constraint length
        
    Returns:
        Next state
    """
    # Shift in new bit, drop oldest bit
    next_state = ((state << 1) | input_bit) & ((1 << (K - 1)) - 1)
    return next_state


def remove_tail_bits(bits: np.ndarray, tail_len: int = 4) -> np.ndarray:
    """Remove tail bits from decoded sequence.
    
    Args:
        bits: Bit sequence with tail bits at the end
        tail_len: Number of tail bits to remove
        
    Returns:
        Bit sequence without tail bits
    """
    if len(bits) <= tail_len:
        return np.array([], dtype=np.uint8)
    return bits[:-tail_len]


def fire_decode_224(encoded_bits: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Decode FIRE-encoded block and verify parity.
    
    Args:
        encoded_bits: 224-bit encoded block (184 info + 40 parity)
        
    Returns:
        Tuple of (184-bit info, parity_valid)
    """
    assert len(encoded_bits) == 224, f"Expected 224 bits, got {len(encoded_bits)}"
    
    # Split into info and parity
    info_bits = encoded_bits[:184]
    received_parity = encoded_bits[184:]
    
    # Recompute parity using the encoding function
    from .encoding import fire_encode_184
    reencoded = fire_encode_184(info_bits)
    computed_parity = reencoded[184:]
    
    # Check if parity matches
    parity_valid = np.array_equal(received_parity, computed_parity)
    
    return info_bits, parity_valid


def decode_bcch_pipeline(data_bursts: list) -> Tuple[np.ndarray, bool]:
    """Complete BCCH decoding pipeline: deinterleave → Viterbi → FIRE decode.
    
    Args:
        data_bursts: List of 4 x 114-bit data arrays from normal bursts
        
    Returns:
        Tuple of (184-bit info, decoding_valid)
    """
    # 1. Deinterleave
    bits456 = deinterleave_4x114_to_456(data_bursts)
    
    # 2. Viterbi decode
    decoded_228 = viterbi_decode(bits456, generators=CONV_GENERATORS, K=CONV_K)
    
    # 3. Remove tail bits
    bits224 = remove_tail_bits(decoded_228, tail_len=4)
    
    # 4. FIRE decode and verify
    info_bits, parity_valid = fire_decode_224(bits224)
    
    return info_bits, parity_valid


def decode_sch_39bits(sch39: np.ndarray) -> Tuple[Optional[int], Optional[int], bool]:
    """Decode SCH 39-bit information block.
    
    Args:
        sch39: 39-bit SCH data (INFO(25) | CRC(10) | TAIL(4))
        
    Returns:
        Tuple of (bsic, fn, crc_valid)
        Returns (None, None, False) if CRC check fails
    """
    assert len(sch39) == 39, f"Expected 39 bits, got {len(sch39)}"
    
    # Extract fields
    info25 = sch39[:25]
    received_crc = sch39[25:35]
    # tail bits at sch39[35:] are not used for decoding
    
    # Verify CRC
    from .burst_builder import compute_sch_crc10
    computed_crc = compute_sch_crc10(info25)
    crc_valid = np.array_equal(received_crc, computed_crc)
    
    if not crc_valid:
        return None, None, False
    
    # Decode BSIC and frame number
    bsic_bits = info25[:6]
    t1_bits = info25[6:17]
    t2_bits = info25[17:22]
    t3p_bits = info25[22:25]
    
    # Convert to integers (MSB first)
    bsic = int(''.join(str(b) for b in bsic_bits), 2)
    T1 = int(''.join(str(b) for b in t1_bits), 2)
    T2 = int(''.join(str(b) for b in t2_bits), 2)
    T3p = int(''.join(str(b) for b in t3p_bits), 2)
    
    # Reconstruct frame number (simplified, exact formula depends on context)
    # This is a simplified reconstruction
    fn = T1 * 26 * 51 + T2 + T3p * 10
    
    return bsic, fn, crc_valid
