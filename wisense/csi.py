"""Raw ESP32 CSI -> per-subcarrier amplitude, exactly as the paper derives it.

Transcribed from the authors' Scripts/CSI.py and Scripts/imf_generator.py
(Armenta-Garcia et al., Sensors 2025, doi:10.3390/s25196220), vectorised.
Every choice below is theirs; where their code and their paper disagree, the
code wins, because the code is what produced the published numbers.

RAW FORMAT
----------
One line per received packet: a millisecond timestamp, then the ESP32's
128-byte LLTF CSI buffer as 64 (imaginary, real) int8 pairs. That ordering is
Espressif's -- imaginary first -- and it is what get_amps() assumes.

THE 52, AND THE 47
------------------
A 20 MHz channel has 64 subcarriers; the DC null (0) and the guard band
(27..37) carry nothing, leaving the 52 the authors' scripts keep. The paper
then says 47 were used, after removing "problematic" ones, without naming
them. Running the published model over every subcarrier names them: raw 1
and 38..41 -- the ones hugging the null and the guard band -- score 19-45%
where every other subcarrier scores 90-97% (tools/probe_published.py).
52 - 5 = 47. They are also the *most variable* subcarriers after min-max
normalisation, because they are mostly noise, so selecting by variance
without excluding them picks the worst input available.

The device port of this file is firmware/wisense/csi_dsp.c. The two are kept
honest by tests/test_csi.py and firmware/tests/test_csi_dsp.c checking the
same frozen reference -- regenerate with `python -m tools.gen_csi_reference`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FS_HZ = 50            # nominal packet rate the dataset was collected at
TIMESTEPS = 850       # first 850 packets of each 20 s recording (~17 s)
N_SUBCARRIERS = 64
CSI_BYTES = 2 * N_SUBCARRIERS

# imf_generator.py: nulls = [0, 27, 28, ..., 37]
NULL_SUBCARRIERS = (0, *range(27, 38))
KEPT_SUBCARRIERS = tuple(i for i in range(N_SUBCARRIERS) if i not in NULL_SUBCARRIERS)
N_KEPT = len(KEPT_SUBCARRIERS)
assert N_KEPT == 52

# Indices into the 52 kept (raw 1, 38, 39, 40, 41). See the module docstring.
PROBLEM_KEPT = (0, 26, 27, 28, 29)
GOOD_KEPT = tuple(k for k in range(N_KEPT) if k not in PROBLEM_KEPT)
assert len(GOOD_KEPT) == 47


def read_raw(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one recording. Returns (timestamps_ms int64 (T,), csi int8-valued int16 (T, 128))."""
    data = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)   # a 1-packet file stays 2-D
    if data.ndim != 2 or data.shape[1] != 1 + CSI_BYTES:
        raise ValueError(f"{path}: expected {1 + CSI_BYTES} columns, got {data.shape}")
    return data[:, 0], data[:, 1:].astype(np.int16)


def amplitude(csi: np.ndarray) -> np.ndarray:
    """(T, 128) interleaved imag/real -> (T, 64) amplitude."""
    csi = np.asarray(csi, dtype=np.float64)
    im, re = csi[..., 0::2], csi[..., 1::2]
    return np.nan_to_num(np.sqrt(im * im + re * re), nan=0.0)


def preprocess(csi: np.ndarray, timesteps: int = TIMESTEPS) -> np.ndarray:
    """Raw recording -> (timesteps, 52) amplitude: truncate, magnitude, drop nulls.

    The paper truncates rather than resamples: packets are taken in arrival
    order, so a recording with packet loss covers slightly more wall-clock
    time than one without. That is a property of the dataset, kept as is.
    """
    if len(csi) < timesteps:
        raise ValueError(f"recording has {len(csi)} packets, need {timesteps}")
    return amplitude(csi[:timesteps])[:, KEPT_SUBCARRIERS]


def minmax(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Per-column min-max scaling over time (axis -2). Constant columns -> lo.

    Matches CSI.normalize(), including its treatment of a flat subcarrier.
    """
    x = np.asarray(x, dtype=np.float64)
    mn = x.min(axis=-2, keepdims=True)
    rng = x.max(axis=-2, keepdims=True) - mn
    flat = rng == 0
    out = (hi - lo) * (x - mn) / np.where(flat, 1.0, rng) + lo
    return np.where(flat, lo, out)


def variance_rank(amp: np.ndarray) -> np.ndarray:
    """Subcarrier indices of one (T, S) recording, most variable first.

    CSI.variance_selection(): variance of the min-max normalised amplitude.
    Ties break toward the lower index, so host and device agree exactly.
    """
    var = minmax(amp).var(axis=-2)
    return np.argsort(-var, kind="stable")
