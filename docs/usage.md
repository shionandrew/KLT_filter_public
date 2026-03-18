# KLT Filter — Usage Guide

A Karhunen–Loève Transform (KLT) based spatial-polariametric filter for interferometric beamforming. The filter suppresses foreground/RFI contamination by solving a generalized eigenvalue problem between a signal covariance matrix and a noise/foreground covariance matrix, then projects the data onto the cleanest mode.

## Installation

```bash
# Core (instrument-agnostic)
pip install .

# With CHIME-specific utilities
pip install ".[chime]"
```

## Package structure

```
klt_filter/
├── klt_beamformer.py    # Core beamformer class (no instrument dependencies)
├── math_core.py         # Pure-math helpers (covariance matrices, KL filter)
├── chime_data_utils.py  # CHIME-specific: calibration, fringestopping, input reordering
└── __init__.py
```

- **`klt_beamformer`** and **`math_core`** can be used with any interferometer. They only require numpy, scipy, and h5py.
- **`chime_data_utils`** requires `ch_util`, `baseband_analysis`, and `beam_model` (CHIME-internal packages).

---

## Quick start (CHIME)

```python
from klt_filter import KLTBeamformer
from klt_filter.chime_data_utils import (
    reorder_inputs,
    apply_gains,
    load_gains,
    compute_fringestop_phases,
    get_good_inputs,
    get_position_from_equatorial,
)
from baseband_analysis.core.bbdata import BBData

# Load data
data = BBData.from_acq_h5(list_of_h5_files)

# CHIME-specific preprocessing
correlator_inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
reordered_inputs, prod_map = reorder_inputs(data, correlator_inputs)
apply_gains(data, gain_file, correlator_inputs)
gains = load_gains(gain_file, data)  # (nfreq, ninput)

# Create the beamformer
beamformer = KLTBeamformer(
    data=data,
    ras=ras,                   # (npointing,) array of RA values
    decs=decs,                 # (npointing,) array of Dec values
    source_names=source_names, # list of str, or None
    obs=backend.obs,           # observatory object
    frame_starts=frame_starts, # (nfreq,) array, or None
    frame_stops=frame_stops,   # (nfreq,) array, or None
)

# Run (with parallel frequency processing)
beamformer(
    freq_indices=range(len(data.freq)),
    gains=gains,
    fringestop_func=lambda fi: compute_fringestop_phases(
        data, fi, ras, decs, reordered_inputs, backend.obs, backend.static_delays,
    ),
    good_inputs_func=lambda fi: get_good_inputs(gains[fi], reordered_inputs),
    output="tiedbeam_baseband",  # or "tiedbeam_visibilities"
    clean=True,
    n_workers=4,                 # number of threads for parallel frequency processing
)

# Write pointing metadata
beamformer.write_tiedbeam_locations(
    get_position_func=get_position_from_equatorial,
)

# Save
data.save("output.h5")
```

---

## Using with a non-CHIME instrument

The core `KLTBeamformer` class has no instrument-specific dependencies. You need to provide:

1. **A data object** that behaves like an h5py group:
   - `data["baseband"][freq_index]` -> `(ninput, ntime)` complex array
   - `data["baseband"].shape` -> `(nfreq, ninput, ntime)`
   - `data["baseband"].dtype` -> complex dtype
   - `data.freq` -> `(nfreq,)` array of frequencies
   - `data.create_dataset(name, shape=..., dtype=...)` -> new dataset
   - `data.keys()` -> list of dataset names

2. **A fringestop function** `fringestop_func(freq_index)` that returns a `(npointing, ninput)` complex array of phases. This encodes the geometric delay model for your instrument.

3. **A good-inputs function** `good_inputs_func(freq_index)` that returns `[x_pol_indices, y_pol_indices]` — integer arrays selecting which inputs to use per polarisation.

4. **Gains** as a `(nfreq, ninput)` complex array (already applied to the data; also passed to the beamformer for input flagging via `good_inputs_func`).

### Example

```python
import numpy as np
from klt_filter import KLTBeamformer

# Your instrument's data (h5py file or compatible object)
data = ...

# Your instrument's delay model
def my_fringestop(freq_index):
    """Return (npointing, ninput) complex fringestop phases."""
    delays = compute_geometric_delays(freq_index)  # your function
    return np.exp(2j * np.pi * data.freq[freq_index] * delays)

# Your instrument's input flagging
def my_good_inputs(freq_index):
    """Return [x_pol_indices, y_pol_indices]."""
    good = np.where(np.abs(gains[freq_index]) > 0)[0]
    x_idx = good[::2]  # example: even indices are X-pol
    y_idx = good[1::2]  # odd indices are Y-pol
    return [x_idx, y_idx]

beamformer = KLTBeamformer(
    data=data,
    ras=np.array([180.0]),    # single pointing
    decs=np.array([45.0]),
    source_names=["my_source"],
    obs=my_observatory,
    frame_starts=frame_starts,
    frame_stops=frame_stops,
)

beamformer(
    freq_indices=range(nfreq),
    gains=gains,
    fringestop_func=my_fringestop,
    good_inputs_func=my_good_inputs,
    output="tiedbeam_baseband",
    clean=True,
    n_workers=4,
)
```

---

## API reference

### `KLTBeamformer`

#### `__init__(data, ras, decs, source_names, obs, frame_starts=None, frame_stops=None)`

| Parameter | Type | Description |
|---|---|---|
| `data` | h5py-like | Dataset with `"baseband"` key, `.freq`, `.input`, `.create_dataset()` |
| `ras` | `(npointing,)` float array | Right ascension of each pointing |
| `decs` | `(npointing,)` float array | Declination of each pointing |
| `source_names` | list of str or None | Name for each pointing |
| `obs` | object | Observatory/telescope position |
| `frame_starts` | array or None | Per-frequency signal start samples |
| `frame_stops` | array or None | Per-frequency signal stop samples |

#### `__call__(freq_indices, gains, fringestop_func, good_inputs_func, output, clean, n_workers)`

| Parameter | Type | Description |
|---|---|---|
| `freq_indices` | list of int | Frequency channels to process |
| `gains` | `(nfreq, ninput)` complex | Calibration gains |
| `fringestop_func` | callable | `f(freq_index)` -> `(npointing, ninput)` complex. Must be thread-safe if `n_workers > 1`. |
| `good_inputs_func` | callable | `f(freq_index)` -> `[x_indices, y_indices]`. Must be thread-safe if `n_workers > 1`. |
| `output` | str | `"tiedbeam_baseband"` or `"tiedbeam_visibilities"` |
| `clean` | bool | Apply KL filter (True) or plain beamform (False) |
| `n_workers` | int | Number of threads for parallel frequency processing (default 1) |

#### `write_tiedbeam_locations(get_position_func=None)`

Writes a `"tiedbeam_locations"` dataset with RA, Dec, position, pol, and source name for each beam. If `get_position_func` is None, `x_400MHz` and `y_400MHz` are set to NaN.

---

### Standalone math functions (`math_core`)

These can be used independently for custom pipelines:

```python
from klt_filter import form_S_from_phase, form_F_from_vis, KL_filter
# or
from klt_filter.math_core import form_S_from_phase, form_F_from_vis, KL_filter
```

| Function | Description |
|---|---|
| `form_S_from_phase(phase)` | Signal covariance from `(ninput,)` fringestop phases |
| `form_F_from_vis(data, start, stop)` | Noise covariance from `(ninput, ntime)` visibility data |
| `form_M_from_V(data)` | Outer-product visibility matrix |
| `KL_filter(S, F, data)` | Solve generalised eigenvalue problem, return cleaned data |
| `apply_pol_covariances(S, N_x, ...)` | Scale S matrix blocks by polarisation cross-powers |

---

### `chime_data_utils`

| Function | Description |
|---|---|
| `reorder_inputs(data, correlator_inputs, reference_feed)` | Reorder inputs to match BBData ordering |
| `load_gains(gain_file, data)` | Load + reorder gains from HDF5 |
| `apply_gains(data, gain_file, correlator_inputs)` | Apply calibration in-place |
| `compute_fringestop_phases(data, freq_index, ras, decs, ...)` | Fringestop phases for one channel |
| `get_good_inputs(gains_freq, reordered_inputs)` | Non-zero gain inputs split by pol |
| `incoherent_dedisp_raw(data, DM, t_ref, ...)` | Incoherent dedispersion in-place |
| `undo_incoherent_dedisp_raw(data)` | Reverse dedispersion |
| `get_position_from_equatorial(ras, decs, ctime)` | Beam-model position wrapper |
