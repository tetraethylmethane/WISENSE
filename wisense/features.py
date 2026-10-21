"""(850, 52) amplitude -> the (850, 1) series the model sees.

The published models take one subcarrier at a time, and were trained with
every subcarrier of every recording as a separate sample -- the paper's
"all subcarriers" configuration taking 10,904 ms is exactly 47 x 232 ms, i.e.
the single-subcarrier model run 47 times. tools/probe_published.py confirms
it: the authors' model scores 90-97% on any one of the 47 good subcarriers.

So there are three ways to score a recording, all reported by evaluate.py:

  per_subcarrier  every good subcarrier scored independently, accuracy
                  averaged. This is the paper's headline metric (their
                  published model scores 0.9243 here -- the paper's 92.43%).
  top_variance    one subcarrier per recording, the most variable of the 47
                  after min-max normalisation. What the device runs: one
                  inference per window. Implemented in csi_dsp.c.
  vote            all 47 classified, majority vote. 47x the latency.
"""

from __future__ import annotations

import numpy as np

from wisense.csi import GOOD_KEPT, variance_rank

N_GOOD = len(GOOD_KEPT)


def good(x: np.ndarray) -> np.ndarray:
    """(..., T, 52) -> (..., T, 47)."""
    return np.asarray(x)[..., list(GOOD_KEPT)]


def per_subcarrier(x: np.ndarray) -> np.ndarray:
    """(N, T, 52) -> (N * 47, T): recording-major, subcarrier-minor."""
    g = good(x)
    return np.ascontiguousarray(g.transpose(0, 2, 1).reshape(-1, g.shape[1])).astype(np.float32)


def top_variance_index(x: np.ndarray) -> np.ndarray:
    """(N, T, 52) -> (N,) index into the 52 of each recording's chosen subcarrier."""
    g = good(x)
    return np.array([GOOD_KEPT[variance_rank(r)[0]] for r in g])


def select(x: np.ndarray) -> np.ndarray:
    """(N, T, 52) -> (N, T) float32: the device's single-subcarrier input."""
    x = np.asarray(x)
    return x[np.arange(len(x)), :, top_variance_index(x)].astype(np.float32)
