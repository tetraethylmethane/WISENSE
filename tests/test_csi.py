"""CSI preprocessing -- the transform both the host and device must agree on."""

import numpy as np
import pytest

from wisense.csi import (KEPT_SUBCARRIERS, N_KEPT, NULL_SUBCARRIERS, amplitude, minmax,
                         preprocess, variance_rank)
from wisense.features import select


def test_kept_subcarriers_exclude_dc_and_guard_band():
    assert N_KEPT == 52
    assert 0 not in KEPT_SUBCARRIERS
    assert not set(range(27, 38)) & set(KEPT_SUBCARRIERS)
    assert len(NULL_SUBCARRIERS) + N_KEPT == 64


def test_amplitude_reads_imag_real_pairs():
    raw = np.zeros((1, 128), dtype=np.int16)
    raw[0, 2], raw[0, 3] = 3, -4     # subcarrier 1
    assert amplitude(raw)[0, 1] == pytest.approx(5.0)
    assert amplitude(raw)[0, 0] == 0.0


def test_preprocess_truncates_and_drops_nulls():
    raw = np.random.default_rng(0).integers(-20, 20, (900, 128))
    out = preprocess(raw)
    assert out.shape == (850, 52)
    assert np.allclose(out, amplitude(raw[:850])[:, list(KEPT_SUBCARRIERS)])


def test_preprocess_rejects_short_recording():
    with pytest.raises(ValueError):
        preprocess(np.zeros((849, 128)))


def test_minmax_flat_column_is_lo_not_nan():
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    out = minmax(x)
    assert np.allclose(out[:, 0], [0, 0.5, 1])
    assert np.allclose(out[:, 1], 0)


def test_variance_rank_is_scale_invariant_and_tie_stable():
    t = np.linspace(0, 1, 850)[:, None]
    amp = np.hstack([np.full((850, 1), 7.0), 100 * t, t ** 4, np.sin(20 * t)])
    rank = variance_rank(amp)
    # min-max normalisation removes scale: the ramp beats a 100x larger flat line
    assert rank[-1] == 0
    flat = np.ones((850, 52))
    assert variance_rank(flat)[0] == 0   # all tied -> lowest index, as the C port does


def test_select_per_recording_picks_each_recordings_own_subcarrier():
    rng = np.random.default_rng(1)
    x = np.ones((2, 850, 52))
    x[0, :, 5] += rng.normal(size=850)
    x[1, :, 40] += rng.normal(size=850)
    out = select(x)
    assert np.allclose(out[0], x[0, :, 5]) and np.allclose(out[1], x[1, :, 40])


def test_per_subcarrier_expands_good_47_recording_major():
    from wisense.csi import GOOD_KEPT
    from wisense.features import per_subcarrier

    x = np.arange(2 * 850 * 52, dtype=np.float32).reshape(2, 850, 52)
    out = per_subcarrier(x)
    assert out.shape == (2 * 47, 850)
    assert np.array_equal(out[0], x[0, :, GOOD_KEPT[0]])
    assert np.array_equal(out[47 + 3], x[1, :, GOOD_KEPT[3]])


def test_select_never_picks_a_problematic_subcarrier():
    from wisense.csi import PROBLEM_KEPT
    from wisense.features import top_variance_index

    x = np.ones((1, 850, 52))
    x[0, ::2, 26] = 100.0                  # raw 38: maximal variance, but excluded
    x[0, ::10, 4] = 2.0                    # raw 5: modest, but real
    assert top_variance_index(x)[0] == 4
    assert 26 in PROBLEM_KEPT
