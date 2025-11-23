"""GSM System Information Message Encoding/Decoding

This module provides basic encoding and decoding for GSM System Information
messages that are carried on BCCH. It implements a simplified version of
GSM 04.08 Layer 3 messages, focusing on System Information Type 3.

System Information Type 3 contains:
- Cell Identity (CI) - 16 bits
- Location Area Code (LAC) - 16 bits  
- Control Channel Description including ARFCN
- Cell Options
- Cell Selection Parameters
- RACH Control Parameters
- Neighbor Cell Description (list of ARFCNs)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


def encode_system_information_type3(
    cell_identity: int = 0,
    location_area_code: int = 0,
    arfcn: int = 0,
    neighbor_cells: Optional[List[int]] = None
) -> np.ndarray:
    """Encode a simplified System Information Type 3 message.
    
    Args:
        cell_identity: Cell Identity (0-65535, 16 bits) - represents base station ID
        location_area_code: Location Area Code (0-65535, 16 bits)
        arfcn: Absolute Radio Frequency Channel Number (0-1023, 10 bits)
        neighbor_cells: List of neighbor cell ARFCNs (up to 16, each 10 bits)
        
    Returns:
        184-bit array for BCCH information payload
    """
    if neighbor_cells is None:
        neighbor_cells = []
    
    # Limit neighbor cells to 16
    neighbor_cells = neighbor_cells[:16]
    
    # Initialize 184-bit array
    info_bits = np.zeros(184, dtype=np.uint8)
    
    # Byte 0: Protocol Discriminator (0x06 for RR management) + Skip Indicator
    # For simplicity, using 0x06 for Radio Resource management
    info_bits[0:8] = _int_to_bits(0x06, 8)
    
    # Byte 1: Message Type (0x1B for System Information Type 3)
    info_bits[8:16] = _int_to_bits(0x1B, 8)
    
    # Bytes 2-3: Cell Identity (16 bits)
    info_bits[16:32] = _int_to_bits(cell_identity, 16)
    
    # Bytes 4-5: Location Area Code (16 bits)
    info_bits[32:48] = _int_to_bits(location_area_code, 16)
    
    # Byte 6-7: Control Channel Description
    # Simplified: ARFCN (10 bits) + padding
    info_bits[48:58] = _int_to_bits(arfcn, 10)
    info_bits[58:64] = 0  # Padding and other control fields
    
    # Byte 8: Cell Options (simplified)
    info_bits[64:72] = _int_to_bits(0x00, 8)
    
    # Bytes 9-10: Cell Selection Parameters (simplified)
    info_bits[72:88] = 0
    
    # Bytes 11-12: RACH Control Parameters (simplified)
    info_bits[88:104] = 0
    
    # Bytes 13+: Neighbor Cell Description
    # Format: number of neighbors (4 bits) + neighbor ARFCNs (10 bits each)
    num_neighbors = len(neighbor_cells)
    info_bits[104:108] = _int_to_bits(num_neighbors, 4)
    
    # Add neighbor cell ARFCNs
    bit_offset = 108
    for i, neighbor_arfcn in enumerate(neighbor_cells):
        if bit_offset + 10 <= 184:
            info_bits[bit_offset:bit_offset+10] = _int_to_bits(neighbor_arfcn, 10)
            bit_offset += 10
        else:
            break
    
    # Remaining bits are padding
    # info_bits[bit_offset:] = 0  # Already initialized to 0
    
    return info_bits


def decode_system_information_type3(info_bits: np.ndarray) -> Dict:
    """Decode a System Information Type 3 message.
    
    Args:
        info_bits: 184-bit information array
        
    Returns:
        Dictionary with decoded fields:
        - protocol_discriminator: Protocol Discriminator value
        - message_type: Message Type value
        - cell_identity: Cell Identity (base station ID)
        - location_area_code: Location Area Code
        - arfcn: ARFCN of this cell
        - neighbor_cells: List of neighbor cell ARFCNs
    """
    if len(info_bits) != 184:
        return {
            'valid': False,
            'error': f'Expected 184 bits, got {len(info_bits)}'
        }
    
    result = {'valid': True}
    
    # Byte 0: Protocol Discriminator
    result['protocol_discriminator'] = _bits_to_int(info_bits[0:8])
    
    # Byte 1: Message Type
    result['message_type'] = _bits_to_int(info_bits[8:16])
    
    # Bytes 2-3: Cell Identity
    result['cell_identity'] = _bits_to_int(info_bits[16:32])
    
    # Bytes 4-5: Location Area Code
    result['location_area_code'] = _bits_to_int(info_bits[32:48])
    
    # Bytes 6-7: ARFCN (first 10 bits)
    result['arfcn'] = _bits_to_int(info_bits[48:58])
    
    # Bytes 13+: Neighbor Cell Description
    num_neighbors = _bits_to_int(info_bits[104:108])
    neighbor_cells = []
    
    bit_offset = 108
    for i in range(min(num_neighbors, 16)):
        if bit_offset + 10 <= 184:
            neighbor_arfcn = _bits_to_int(info_bits[bit_offset:bit_offset+10])
            neighbor_cells.append(neighbor_arfcn)
            bit_offset += 10
        else:
            break
    
    result['neighbor_cells'] = neighbor_cells
    
    return result


def format_system_information(info_bits: np.ndarray) -> str:
    """Format System Information for human-readable display.
    
    Args:
        info_bits: 184-bit information array
        
    Returns:
        Formatted string with System Information breakdown
    """
    decoded = decode_system_information_type3(info_bits)
    
    if not decoded.get('valid', False):
        return f"Invalid System Information: {decoded.get('error', 'Unknown error')}"
    
    output = []
    output.append("")
    output.append("=" * 60)
    output.append("System Information Type 3 (BCCH)")
    output.append("=" * 60)
    output.append("")
    
    # Check if this looks like a System Information Type 3 message
    if decoded['message_type'] == 0x1B:
        output.append(f"Message Type: 0x{decoded['message_type']:02X} (System Information Type 3)")
    else:
        output.append(f"Message Type: 0x{decoded['message_type']:02X}")
        output.append("(Note: Expected 0x1B for System Information Type 3)")
    
    output.append("")
    output.append("Cell Information:")
    output.append(f"  Cell Identity (Base Station ID): {decoded['cell_identity']}")
    output.append(f"  Location Area Code: {decoded['location_area_code']}")
    output.append(f"  ARFCN: {decoded['arfcn']}")
    output.append("")
    
    # Display neighbor cells
    neighbor_cells = decoded.get('neighbor_cells', [])
    if neighbor_cells:
        output.append(f"Neighbor Cells ({len(neighbor_cells)}):")
        for i, arfcn in enumerate(neighbor_cells):
            output.append(f"  {i+1}. ARFCN: {arfcn}")
    else:
        output.append("Neighbor Cells: None")
    
    output.append("")
    output.append("=" * 60)
    output.append("")
    
    return '\n'.join(output)


def _int_to_bits(value: int, num_bits: int) -> np.ndarray:
    """Convert integer to bit array (MSB first).
    
    Args:
        value: Integer value to convert
        num_bits: Number of bits in output
        
    Returns:
        Bit array with MSB first
    """
    bits = np.array([(value >> i) & 1 for i in reversed(range(num_bits))], dtype=np.uint8)
    return bits


def _bits_to_int(bits: np.ndarray) -> int:
    """Convert bit array (MSB first) to integer.
    
    Args:
        bits: Bit array with MSB first
        
    Returns:
        Integer value
    """
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value
