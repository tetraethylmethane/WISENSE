"""Augmentation and the split. A leak here trains fine and reports a great number."""

from math import comb

import numpy as np

from wisense.augment import N_MODES, recombine, synthesize
from wisense.data import SUBJECTS, TEST_SUBJECTS, TRAIN_SUBJECTS


def test_split_is_by_subject_and_complete():
    assert not set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS)
    assert set(TRAIN_SUBJECTS) | set(TEST_SUBJECTS) == set(SUBJECTS)
    assert len(SUBJECTS) == 22 and len(TEST_SUBJECTS) == 6


def test_recombine_takes_even_modes_from_a_and_odd_from_b():
    a = np.arange(N_MODES)[:, None] * np.ones((1, 4))
    b = 100 + a
    # even rows of a: 0+2+4+6 = 12; odd rows of b: 101+103+105+107 = 416
    assert np.allclose(recombine(a, b), 12 + 416)


def test_synthesize_pairs_only_within_subject_and_activity():
    rng = np.random.default_rng(0)
    subject = np.array(["s1"] * 5 + ["s2"] * 4 + ["s1"] * 3)
    y = np.array([0] * 5 + [0] * 4 + [1] * 3)
    modes = rng.normal(size=(len(y), N_MODES, 10)).astype(np.float32)
    # tag each recording so we can tell which ones were mixed
    for i in range(len(y)):
        modes[i, 0] = i
        modes[i, 1] = 1000 * i
    sx, sy, ss = synthesize(modes, y, subject)
    assert len(sx) == comb(5, 2) + comb(4, 2) + comb(3, 2)
    groups = {(s, l): set(np.flatnonzero((subject == s) & (y == l))) for s, l in zip(subject, y)}
    for x, label, subj in zip(sx, sy, ss):
        # mode 0 (even) carries i, mode 1 (odd) carries 1000*j; others are noise
        members = groups[(subj, label)]
        assert any(np.allclose(x - modes[i, 2::2].sum(0) - modes[j, 3::2].sum(0), i + 1000 * j)
                   for i in members for j in members if i < j)


def test_paper_synthetic_count_implies_test_subjects_were_augmented():
    """The arithmetic behind the leak claim in data.py."""
    per_group = comb(15, 2)
    all_subjects = len(SUBJECTS) * 5 * per_group
    train_only = len(TRAIN_SUBJECTS) * 5 * per_group
    paper = 11_513
    assert abs(all_subjects - paper) < abs(train_only - paper)
    assert abs(all_subjects - paper) / paper < 0.01


def test_synthesize_all_never_mixes_subcarriers():
    from wisense.augment import synthesize_all

    n, s = 4, 3
    modes = np.zeros((n, s, N_MODES, 5), dtype=np.float32)
    for k in range(s):
        modes[:, k] = k * 1000          # every mode of subcarrier k is k*1000
    y = np.zeros(n, int)
    subject = np.array(["s1"] * n)
    sx, sy = synthesize_all(modes, y, subject)
    assert len(sx) == s * comb(n, 2)
    # 4 even + 4 odd modes, all from one subcarrier -> 8 * k * 1000
    per = comb(n, 2)
    for k in range(s):
        assert np.allclose(sx[k * per:(k + 1) * per], 8 * k * 1000)


def test_spread_subcarriers_are_good_and_distinct():
    from wisense.augment import spread_subcarriers
    from wisense.csi import GOOD_KEPT

    for n in (1, 8, 47):
        scs = spread_subcarriers(n)
        assert len(set(scs)) == n and set(scs) <= set(GOOD_KEPT)
