# GSM Waveform Generator

A Python library for generating GSM broadcast channel waveforms (BCCH, FCCH, SCH) conforming to 3GPP/ETSI GSM specifications (TS 45.002, TS 45.003).

## Features

- **Complete GSM Baseband Implementation**
  - FIRE code encoding (40-bit parity for 184-bit information)
  - Convolutional encoding (rate 1/2, K=5)
  - Block interleaving (456 bits → 4×114 bits)
  - Normal Burst, FCCH, and SCH burst construction
  - GMSK modulation (BT=0.3)

- **Specification Compliant**
  - Follows GSM 05.02 and 05.03 standards
  - Proper training sequence codes (TSC 0-7)
  - Correct synchronization channel structure
  - Standard symbol rate (270.833 ksps)

- **SDR Ready**
  - Generates complex IQ samples compatible with GNU Radio
  - Ready for HackRF One, USRP, and other SDR platforms
  - Supports standard file formats for SDR tools

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from gsm_waveform import (
    make_bcch_encoded_456,
    interleave_456_to_4x114,
    build_normal_burst,
    build_fcch_burst,
    build_sch_burst,
    get_tsc,
    modulate_burst_sequence,
    write_complex_iq
)

# Create 184-bit information payload
info_bits = np.zeros(184, dtype=np.uint8)

# Encode with FIRE code and convolutional encoding
encoded = make_bcch_encoded_456(info_bits)

# Interleave to 4 bursts
data_bursts = interleave_456_to_4x114(encoded)

# Build bursts
fcch = build_fcch_burst()
sch = build_sch_burst(bsic=10, fn=0)
tsc = get_tsc(0)
normal_bursts = [build_normal_burst(data, tsc) for data in data_bursts]

# Modulate to IQ samples
all_bursts = [fcch, sch] + normal_bursts
iq = modulate_burst_sequence(all_bursts, guard_samples=66)

# Save to file
write_complex_iq("output.cfile", iq)
```

## Examples

See the `examples/` directory for complete examples:

```bash
python examples/generate_bcch_waveform.py
```

This generates a complete BCCH frame with FCCH, SCH, and 4 normal bursts.

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=gsm_waveform tests/
```

## Module Overview

- **`constants.py`** - GSM constants and training sequences
- **`encoding.py`** - FIRE code and convolutional encoding
- **`interleave.py`** - Block interleaving for control channels
- **`burst_builder.py`** - Normal Burst, FCCH, SCH construction
- **`modulator.py`** - GMSK modulation (BT=0.3)
- **`wavefile.py`** - IQ file I/O utilities

## Usage with SDR Tools

### HackRF One

```bash
hackrf_transfer -t output.cfile -f 935000000 -s 2167000 -a 1 -x 20
```

### GNU Radio

Use the File Source block:
- File: `output.cfile`
- Sample Rate: 2167000
- Type: Complex float32

## Legal Notice

**⚠️ WARNING:** Transmitting on GSM frequencies without proper authorization is illegal in most countries. This software is intended for:
- Educational purposes
- Research in controlled environments
- Licensed use with proper attenuation
- Compliance with local regulations

Always use proper RF shielding and attenuation when testing. Check your local laws before transmitting on any frequency.

## Technical Details

### Symbol Rate
- 270.833 ksps (GSM standard)

### Modulation
- GMSK with BT=0.3
- Gaussian filter span: 4 symbols

### Encoding
- FIRE code: (D^23 + 1) × (D^17 + D^3 + 1)
- Convolutional: Rate 1/2, K=5, generators (0o23, 0o35)

### Burst Structure
- Normal Burst: 148 bits (3 tail + 57 data + 1 stealing + 26 TSC + 1 stealing + 57 data + 3 tail)
- FCCH: 148 bits (all zeros)
- SCH: 148 bits (3 tail + 39 data + 64 extended TSC + 39 data + 3 tail)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## References

- 3GPP TS 45.002: Multiplexing and multiple access on the radio path
- 3GPP TS 45.003: Channel coding
- ETSI GSM 05.02 and 05.03 specifications
