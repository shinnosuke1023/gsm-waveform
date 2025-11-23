"""GSM System Information Message Encoding/Decoding

This module implements GSM System Information Type 3 encoding and decoding
according to GSM 04.08 specification (3GPP TS 24.008).

System Information Type 3 message structure (GSM 04.08 Section 9.1.35):
- Protocol Discriminator (4 bits) + Skip Indicator (4 bits)
- Message Type (8 bits)
- Cell Identity (16 bits)
- Location Area Identification (40 bits):
  - Mobile Country Code (MCC) - 3 digits BCD
  - Mobile Network Code (MNC) - 2-3 digits BCD
  - Location Area Code (LAC) - 16 bits
- Control Channel Description (24 bits)
- Cell Options (BCCH) (8 bits)
- Cell Selection Parameters (16 bits)
- RACH Control Parameters (24 bits)
- SI 3 Rest Octets (variable length)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


def encode_system_information_type3(
    cell_identity: int = 0,
    location_area_code: int = 0,
    mobile_country_code: int = 1,
    mobile_network_code: int = 1,
    arfcn: int = 0,
    bsic: int = 0,
    # Control Channel Description parameters
    att: bool = True,
    bs_ag_blks_res: int = 1,
    ccch_conf: int = 1,
    bs_pa_mfrms: int = 2,
    t3212: int = 0,
    # Cell Options
    pwrc: bool = False,
    dtx: int = 2,
    radio_link_timeout: int = 4,
    # Cell Selection Parameters
    cell_reselect_hysteresis: int = 2,
    ms_txpwr_max_cch: int = 0,
    rxlev_access_min: int = 0,
    neci: bool = True,
    acs: bool = False,
    # RACH Control Parameters
    max_retrans: int = 1,
    tx_integer: int = 3,
    cell_bar_access: bool = False,
    re: bool = False,
    acc: int = 0xFFFF,
    # Neighbor cells
    neighbor_cells: Optional[List[int]] = None
) -> np.ndarray:
    """Encode System Information Type 3 message according to GSM 04.08.
    
    Args:
        cell_identity: Cell Identity (0-65535, 16 bits)
        location_area_code: Location Area Code (0-65535, 16 bits)
        mobile_country_code: MCC (0-999, 3 BCD digits)
        mobile_network_code: MNC (0-999, 2-3 BCD digits)
        arfcn: ARFCN (0-1023, 10 bits)
        bsic: Base Station Identity Code (0-63, 6 bits)
        att: Attach/detach allowed flag
        bs_ag_blks_res: Number of blocks reserved for AGCH (0-7)
        ccch_conf: CCCH configuration (0-7)
        bs_pa_mfrms: BS paging multiframes (0-9)
        t3212: Periodic location update timer (0-255, in deci-hours)
        pwrc: Power control indicator
        dtx: Discontinuous transmission (0-3)
        radio_link_timeout: Radio link timeout (0-15)
        cell_reselect_hysteresis: Cell reselect hysteresis (0-7, 2dB steps)
        ms_txpwr_max_cch: MS max TX power level on CCCH (0-31)
        rxlev_access_min: Minimum RX level for access (0-63)
        neci: New establishment causes indicator
        acs: Additional reselect parameter indicator
        max_retrans: Maximum retransmissions (0-3)
        tx_integer: Number of slots to spread transmission (0-15)
        cell_bar_access: Cell barred indicator
        re: Call reestablishment allowed
        acc: Access control class bitmap (16 bits)
        neighbor_cells: List of neighbor cell ARFCNs
        
    Returns:
        184-bit array for BCCH information payload
    """
    if neighbor_cells is None:
        neighbor_cells = []
    
    # Initialize 184-bit array (23 bytes)
    info_bits = np.zeros(184, dtype=np.uint8)
    bit_pos = 0
    
    # Byte 0: Protocol Discriminator (4 bits) + Skip Indicator (4 bits)
    # PD = 0x06 for Radio Resource Management
    # Skip Indicator = 0x0 (no extension)
    info_bits[bit_pos:bit_pos+4] = _int_to_bits(0x0, 4)  # Skip Indicator
    info_bits[bit_pos+4:bit_pos+8] = _int_to_bits(0x6, 4)  # PD
    bit_pos += 8
    
    # Byte 1: Message Type (0x1B for System Information Type 3)
    info_bits[bit_pos:bit_pos+8] = _int_to_bits(0x1B, 8)
    bit_pos += 8
    
    # Bytes 2-3: Cell Identity (16 bits, MSB first)
    info_bits[bit_pos:bit_pos+16] = _int_to_bits(cell_identity, 16)
    bit_pos += 16
    
    # Bytes 4-8: Location Area Identification (5 bytes = 40 bits)
    # MCC and MNC in BCD format (GSM 04.08 Section 10.5.1.3)
    # _int_to_bcd_digits returns [ones, tens, hundreds], so reverse for proper order
    mcc_digits = _int_to_bcd_digits(mobile_country_code, 3)
    mcc_digits.reverse()  # Now [hundreds, tens, ones] = [d1, d2, d3]
    mnc_digits = _int_to_bcd_digits(mobile_network_code, 2 if mobile_network_code < 100 else 3)
    mnc_digits.reverse()  # Now [tens, ones] or [hundreds, tens, ones]
    
    # Byte 4: MCC digit 2 (4 bits) + MCC digit 1 (4 bits)
    info_bits[bit_pos:bit_pos+4] = _int_to_bits(mcc_digits[1], 4)
    info_bits[bit_pos+4:bit_pos+8] = _int_to_bits(mcc_digits[0], 4)
    bit_pos += 8
    
    # Byte 5: MNC digit 3 (or 0xF if 2-digit) (4 bits) + MCC digit 3 (4 bits)
    mnc_digit3 = mnc_digits[2] if len(mnc_digits) == 3 else 0xF
    info_bits[bit_pos:bit_pos+4] = _int_to_bits(mnc_digit3, 4)
    info_bits[bit_pos+4:bit_pos+8] = _int_to_bits(mcc_digits[2], 4)
    bit_pos += 8
    
    # Byte 6: MNC digit 2 (4 bits) + MNC digit 1 (4 bits)
    # For 2-digit MNC: mnc_digits = [tens, ones], so d1=tens, d2=ones
    # For 3-digit MNC: mnc_digits = [hundreds, tens, ones], so d1=hundreds, d2=tens
    mnc_d2 = mnc_digits[1] if len(mnc_digits) >= 2 else 0
    mnc_d1 = mnc_digits[0] if len(mnc_digits) >= 1 else 0
    info_bits[bit_pos:bit_pos+4] = _int_to_bits(mnc_d2, 4)
    info_bits[bit_pos+4:bit_pos+8] = _int_to_bits(mnc_d1, 4)
    bit_pos += 8
    
    # Bytes 7-8: Location Area Code (16 bits, MSB first)
    info_bits[bit_pos:bit_pos+16] = _int_to_bits(location_area_code, 16)
    bit_pos += 16
    
    # Bytes 9-11: Control Channel Description (3 bytes = 24 bits)
    # GSM 04.08 Section 10.5.2.11
    # Bit 8: MSCR (MSC Release) - set to 1
    # Bit 7: ATT (Attach/detach allowed)
    # Bit 6-4: BS_AG_BLKS_RES (0-7)
    # Bit 3-1: CCCH-CONF (0-7)
    info_bits[bit_pos] = 1  # MSCR
    info_bits[bit_pos+1] = 1 if att else 0
    info_bits[bit_pos+2:bit_pos+5] = _int_to_bits(bs_ag_blks_res, 3)
    info_bits[bit_pos+5:bit_pos+8] = _int_to_bits(ccch_conf, 3)
    bit_pos += 8
    
    # Byte 10: BS_PA_MFRMS (3 bits) + spare (2 bits) + T3212 MSB (3 bits)
    info_bits[bit_pos:bit_pos+3] = _int_to_bits(bs_pa_mfrms, 3)
    info_bits[bit_pos+3:bit_pos+5] = 0  # spare
    info_bits[bit_pos+5:bit_pos+8] = _int_to_bits((t3212 >> 5) & 0x7, 3)
    bit_pos += 8
    
    # Byte 11: T3212 LSB (5 bits) + spare (3 bits)
    info_bits[bit_pos:bit_pos+5] = _int_to_bits(t3212 & 0x1F, 5)
    info_bits[bit_pos+5:bit_pos+8] = 0  # spare
    bit_pos += 8
    
    # Byte 12: Cell Options (BCCH) - GSM 04.08 Section 10.5.2.3
    # Bit 8: PWRC (Power control indicator)
    # Bit 7-6: DTX (Discontinuous Transmission indicator)
    # Bit 5-1: RADIO-LINK-TIMEOUT
    info_bits[bit_pos] = 1 if pwrc else 0
    info_bits[bit_pos+1:bit_pos+3] = _int_to_bits(dtx, 2)
    info_bits[bit_pos+3:bit_pos+8] = _int_to_bits(radio_link_timeout, 5)
    bit_pos += 8
    
    # Bytes 13-14: Cell Selection Parameters - GSM 04.08 Section 10.5.2.4
    # Byte 13: CELL-RESELECT-HYSTERESIS (3 bits) + MS-TXPWR-MAX-CCH (5 bits)
    info_bits[bit_pos:bit_pos+3] = _int_to_bits(cell_reselect_hysteresis, 3)
    info_bits[bit_pos+3:bit_pos+8] = _int_to_bits(ms_txpwr_max_cch, 5)
    bit_pos += 8
    
    # Byte 14: RXLEV-ACCESS-MIN (6 bits) + NECI (1 bit) + ACS (1 bit)
    info_bits[bit_pos:bit_pos+6] = _int_to_bits(rxlev_access_min, 6)
    info_bits[bit_pos+6] = 1 if neci else 0
    info_bits[bit_pos+7] = 1 if acs else 0
    bit_pos += 8
    
    # Bytes 15-17: RACH Control Parameters - GSM 04.08 Section 10.5.2.29
    # Byte 15: MAX-RETRANS (2 bits) + TX-INTEGER (4 bits) + CELL-BAR-ACCESS (1 bit) + RE (1 bit)
    info_bits[bit_pos:bit_pos+2] = _int_to_bits(max_retrans, 2)
    info_bits[bit_pos+2:bit_pos+6] = _int_to_bits(tx_integer, 4)
    info_bits[bit_pos+6] = 1 if cell_bar_access else 0
    info_bits[bit_pos+7] = 1 if re else 0
    bit_pos += 8
    
    # Bytes 16-17: Access Control Class (16 bits)
    info_bits[bit_pos:bit_pos+16] = _int_to_bits(acc, 16)
    bit_pos += 16
    
    # Bytes 18-23: SI 3 Rest Octets (6 bytes = 48 bits)
    # Custom encoding: ARFCN (10 bits) + neighbor cell list
    # Note: In real GSM 04.08, ARFCN would be implicit or in Cell Channel Description
    # We include it here for completeness and testing
    
    # Encode serving cell ARFCN (10 bits)
    if bit_pos + 10 <= 184:
        info_bits[bit_pos:bit_pos+10] = _int_to_bits(arfcn, 10)
        bit_pos += 10
    
    # Encode neighbor cells
    if neighbor_cells and bit_pos + 4 <= 184:
        # Calculate how many neighbors will fit in remaining space
        available_bits = 184 - bit_pos - 4  # Reserve 4 bits for count
        max_neighbors_that_fit = min(available_bits // 10, len(neighbor_cells), 15)  # Max 15 (4 bits = 0-15)
        
        # Encode neighbor cell count (4 bits)
        info_bits[bit_pos:bit_pos+4] = _int_to_bits(max_neighbors_that_fit, 4)
        bit_pos += 4
        
        # Encode each neighbor ARFCN (10 bits each)
        for i in range(max_neighbors_that_fit):
            if bit_pos + 10 <= 184:
                info_bits[bit_pos:bit_pos+10] = _int_to_bits(neighbor_cells[i], 10)
                bit_pos += 10
            else:
                break
    
    # Remaining bits are padding (already initialized to 0)
    
    return info_bits


def decode_system_information_type3(info_bits: np.ndarray) -> Dict:
    """Decode System Information Type 3 message according to GSM 04.08.
    
    Args:
        info_bits: 184-bit information array
        
    Returns:
        Dictionary with decoded fields including all GSM 04.08 parameters
    """
    if len(info_bits) != 184:
        return {
            'valid': False,
            'error': f'Expected 184 bits, got {len(info_bits)}'
        }
    
    result = {'valid': True}
    bit_pos = 0
    
    # Byte 0: Skip Indicator (4 bits) + Protocol Discriminator (4 bits)
    result['skip_indicator'] = _bits_to_int(info_bits[bit_pos:bit_pos+4])
    result['protocol_discriminator'] = _bits_to_int(info_bits[bit_pos+4:bit_pos+8])
    bit_pos += 8
    
    # Byte 1: Message Type
    result['message_type'] = _bits_to_int(info_bits[bit_pos:bit_pos+8])
    bit_pos += 8
    
    # Bytes 2-3: Cell Identity
    result['cell_identity'] = _bits_to_int(info_bits[bit_pos:bit_pos+16])
    bit_pos += 16
    
    # Bytes 4-8: Location Area Identification
    # Byte 4: MCC digit 2 + MCC digit 1
    mcc_d2 = _bits_to_int(info_bits[bit_pos:bit_pos+4])
    mcc_d1 = _bits_to_int(info_bits[bit_pos+4:bit_pos+8])
    bit_pos += 8
    
    # Byte 5: MNC digit 3 + MCC digit 3
    mnc_d3 = _bits_to_int(info_bits[bit_pos:bit_pos+4])
    mcc_d3 = _bits_to_int(info_bits[bit_pos+4:bit_pos+8])
    bit_pos += 8
    
    # Byte 6: MNC digit 2 + MNC digit 1
    mnc_d2 = _bits_to_int(info_bits[bit_pos:bit_pos+4])
    mnc_d1 = _bits_to_int(info_bits[bit_pos+4:bit_pos+8])
    bit_pos += 8
    
    # Convert BCD to integer
    result['mobile_country_code'] = mcc_d1 * 100 + mcc_d2 * 10 + mcc_d3
    if mnc_d3 == 0xF:
        result['mobile_network_code'] = mnc_d1 * 10 + mnc_d2
    else:
        result['mobile_network_code'] = mnc_d1 * 100 + mnc_d2 * 10 + mnc_d3
    
    # Bytes 7-8: Location Area Code
    result['location_area_code'] = _bits_to_int(info_bits[bit_pos:bit_pos+16])
    bit_pos += 16
    
    # Bytes 9-11: Control Channel Description
    # Byte 9
    result['mscr'] = _bits_to_int(info_bits[bit_pos:bit_pos+1])
    result['att'] = _bits_to_int(info_bits[bit_pos+1:bit_pos+2]) == 1
    result['bs_ag_blks_res'] = _bits_to_int(info_bits[bit_pos+2:bit_pos+5])
    result['ccch_conf'] = _bits_to_int(info_bits[bit_pos+5:bit_pos+8])
    bit_pos += 8
    
    # Byte 10
    result['bs_pa_mfrms'] = _bits_to_int(info_bits[bit_pos:bit_pos+3])
    # Skip spare bits
    t3212_msb = _bits_to_int(info_bits[bit_pos+5:bit_pos+8])
    bit_pos += 8
    
    # Byte 11
    t3212_lsb = _bits_to_int(info_bits[bit_pos:bit_pos+5])
    result['t3212'] = (t3212_msb << 5) | t3212_lsb
    bit_pos += 8
    
    # Byte 12: Cell Options (BCCH)
    result['pwrc'] = _bits_to_int(info_bits[bit_pos:bit_pos+1]) == 1
    result['dtx'] = _bits_to_int(info_bits[bit_pos+1:bit_pos+3])
    result['radio_link_timeout'] = _bits_to_int(info_bits[bit_pos+3:bit_pos+8])
    bit_pos += 8
    
    # Bytes 13-14: Cell Selection Parameters
    # Byte 13
    result['cell_reselect_hysteresis'] = _bits_to_int(info_bits[bit_pos:bit_pos+3])
    result['ms_txpwr_max_cch'] = _bits_to_int(info_bits[bit_pos+3:bit_pos+8])
    bit_pos += 8
    
    # Byte 14
    result['rxlev_access_min'] = _bits_to_int(info_bits[bit_pos:bit_pos+6])
    result['neci'] = _bits_to_int(info_bits[bit_pos+6:bit_pos+7]) == 1
    result['acs'] = _bits_to_int(info_bits[bit_pos+7:bit_pos+8]) == 1
    bit_pos += 8
    
    # Bytes 15-17: RACH Control Parameters
    # Byte 15
    result['max_retrans'] = _bits_to_int(info_bits[bit_pos:bit_pos+2])
    result['tx_integer'] = _bits_to_int(info_bits[bit_pos+2:bit_pos+6])
    result['cell_bar_access'] = _bits_to_int(info_bits[bit_pos+6:bit_pos+7]) == 1
    result['re'] = _bits_to_int(info_bits[bit_pos+7:bit_pos+8]) == 1
    bit_pos += 8
    
    # Bytes 16-17: Access Control Class
    result['acc'] = _bits_to_int(info_bits[bit_pos:bit_pos+16])
    bit_pos += 16
    
    # Bytes 18-23: SI 3 Rest Octets - decode ARFCN and neighbor cells
    # Custom encoding: ARFCN (10 bits) + neighbor cell list
    
    # Decode serving cell ARFCN (10 bits)
    if bit_pos + 10 <= 184:
        result['arfcn'] = _bits_to_int(info_bits[bit_pos:bit_pos+10])
        bit_pos += 10
    else:
        result['arfcn'] = 0
    
    # Decode neighbor cells
    neighbor_cells = []
    if bit_pos + 4 <= 184:
        num_neighbors = _bits_to_int(info_bits[bit_pos:bit_pos+4])
        bit_pos += 4
        
        for i in range(min(num_neighbors, 16)):
            if bit_pos + 10 <= 184:
                neighbor_arfcn = _bits_to_int(info_bits[bit_pos:bit_pos+10])
                neighbor_cells.append(neighbor_arfcn)
                bit_pos += 10
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
    output.append("System Information Type 3 (GSM 04.08)")
    output.append("=" * 60)
    output.append("")
    
    # Check if this looks like a System Information Type 3 message
    if decoded['message_type'] == 0x1B:
        output.append(f"Message Type: 0x{decoded['message_type']:02X} (System Information Type 3)")
    else:
        output.append(f"Message Type: 0x{decoded['message_type']:02X}")
        output.append("(Note: Expected 0x1B for System Information Type 3)")
    
    output.append("")
    output.append("Location Area Identification:")
    output.append(f"  Mobile Country Code (MCC): {decoded['mobile_country_code']}")
    output.append(f"  Mobile Network Code (MNC): {decoded['mobile_network_code']}")
    output.append(f"  Location Area Code (LAC): {decoded['location_area_code']}")
    output.append(f"  Cell Identity (CI): {decoded['cell_identity']}")
    output.append(f"  ARFCN: {decoded.get('arfcn', 'N/A')}")
    output.append("")
    
    output.append("Control Channel Description:")
    output.append(f"  MSCR: {decoded['mscr']}")
    output.append(f"  ATT (Attach/Detach Allowed): {decoded['att']}")
    output.append(f"  BS_AG_BLKS_RES: {decoded['bs_ag_blks_res']}")
    output.append(f"  CCCH-CONF: {decoded['ccch_conf']}")
    output.append(f"  BS_PA_MFRMS: {decoded['bs_pa_mfrms']}")
    output.append(f"  T3212 (Periodic Update Timer): {decoded['t3212']} deci-hours")
    output.append("")
    
    output.append("Cell Options (BCCH):")
    output.append(f"  PWRC (Power Control): {decoded['pwrc']}")
    output.append(f"  DTX: {decoded['dtx']}")
    output.append(f"  Radio Link Timeout: {decoded['radio_link_timeout']}")
    output.append("")
    
    output.append("Cell Selection Parameters:")
    output.append(f"  Cell Reselect Hysteresis: {decoded['cell_reselect_hysteresis']} (x2 dB)")
    output.append(f"  MS TXPWR MAX CCH: {decoded['ms_txpwr_max_cch']}")
    output.append(f"  RXLEV ACCESS MIN: {decoded['rxlev_access_min']}")
    output.append(f"  NECI (New Establishment Causes): {decoded['neci']}")
    output.append(f"  ACS (Additional Reselect Param): {decoded['acs']}")
    output.append("")
    
    output.append("RACH Control Parameters:")
    output.append(f"  Max Retransmissions: {decoded['max_retrans']}")
    output.append(f"  TX-INTEGER: {decoded['tx_integer']}")
    output.append(f"  Cell Bar Access: {decoded['cell_bar_access']}")
    output.append(f"  Call Reestablishment: {decoded['re']}")
    output.append(f"  Access Control Class: 0x{decoded['acc']:04X}")
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


def _int_to_bcd_digits(value: int, num_digits: int) -> List[int]:
    """Convert integer to BCD digits.
    
    Args:
        value: Integer value to convert
        num_digits: Number of digits to return
        
    Returns:
        List of BCD digits (0-9)
    """
    digits = []
    for i in range(num_digits):
        digits.append(value % 10)
        value //= 10
    return digits
