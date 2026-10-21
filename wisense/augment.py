"""EEMD recombination augmentation (paper Section 3.3).

A subcarrier's amplitude is decomposed by ensemble EMD into 7 intrinsic mode
functions plus a residue. A synthetic sample takes the even modes of
recording A and the odd modes of recording B -- same subcarrier, same
subject, same activity -- for every pair A < B, and sums them. 15
repetitions give C(15,2) = 105 synthetic recordings per subject-activity.

Parameters are the authors' (imf_generator.py): EEMD(trials=100), noise seed
5, max_imf=7, so 8 rows including the residue. Recordings that decompose into
fewer modes are zero-padded -- the original reused one buffer across
subcarriers, which left the previous subcarrier's modes in the unused rows.

Departures, and why:

  * By default (--protocol clean) only TRAIN_SUBJECTS are decomposed, which
    closes the leak described in data.py. --protocol paper reproduces the
    authors' scripts exactly, test subjects included, so the paper's numbers
    can be reproduced as they were obtained. Its results are labelled as such.
  * Not every subcarrier. EEMD costs ~5 s per 850-sample series on a laptop
    CPU; all 47 good subcarriers of 1,198 recordings is ~80 CPU-hours. By
    default a fixed, evenly spaced subset of AUG_DEFAULT subcarriers is
    decomposed (~75 min on 11 cores), and --subcarriers 47 does the lot
    overnight. Subcarriers are strongly correlated, so a spread subset keeps
    most of the diversity.

    python -m wisense.augment                    # 8 subcarriers
    python -m wisense.augment --subcarriers 47   # all, ~7-8 h on 12 cores
    python -m wisense.augment --subcarriers 47 --protocol paper   # + test subjects
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

from wisense.csi import GOOD_KEPT

N_MODES = 8          # 7 IMFs + residue
EEMD_TRIALS = 100
EEMD_SEED = 5
AUG_DEFAULT = 8

# clean: synthetic samples from training subjects only (the correct protocol).
# paper: from all 22 subjects, reproducing imf_sample_builder.py exactly --
#        recombined test-subject recordings end up in the training set.
PROTOCOLS = ("clean", "paper")
IMF_CACHE = {"clean": "imfs_train.npz", "paper": "imfs_paper.npz"}


def decompose(signal: np.ndarray, trials: int = EEMD_TRIALS, seed: int = EEMD_SEED) -> np.ndarray:
    """(T,) -> (8, T) EEMD modes, last row the residue, zero-padded."""
    from PyEMD import EEMD

    eemd = EEMD(trials=trials)
    eemd.noise_seed(seed)
    t = np.linspace(0, len(signal), len(signal))
    modes = eemd.eemd(np.asarray(signal, dtype=np.float64), t, max_imf=N_MODES - 1)
    out = np.zeros((N_MODES, len(signal)), dtype=np.float32)
    out[:len(modes)] = modes[:N_MODES]
    return out


def decompose_all(signals: np.ndarray, workers: int | None = None, trials: int = EEMD_TRIALS) -> np.ndarray:
    """(N, T) -> (N, 8, T), in parallel across series."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out = np.empty((len(signals), N_MODES, signals.shape[1]), dtype=np.float32)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, modes in enumerate(pool.map(decompose, signals, [trials] * len(signals), chunksize=8)):
            out[i] = modes
            if i and i % 200 == 0:
                eta = (time.time() - t0) / i * (len(signals) - i)
                print(f"  EEMD {i}/{len(signals)}  eta {eta / 60:.0f} min", flush=True)
    return out


def spread_subcarriers(n: int) -> tuple[int, ...]:
    """n evenly spaced indices into the 52, drawn from the 47 good ones."""
    pick = np.linspace(0, len(GOOD_KEPT) - 1, n).round().astype(int)
    return tuple(GOOD_KEPT[i] for i in pick)


def recombine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Even modes of a + odd modes of b -> one (T,) synthetic signal."""
    return a[0::2].sum(axis=0) + b[1::2].sum(axis=0)


def synthesize(modes: np.ndarray, y: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All within-(subject, activity) pairs for one subcarrier.

    modes (N, 8, T) -> (x (M, T), y (M,), subject (M,)).
    """
    xs, ys, ss = [], [], []
    for subj in np.unique(subject):
        for label in np.unique(y):
            idx = np.flatnonzero((subject == subj) & (y == label))
            for i, j in combinations(idx, 2):
                xs.append(recombine(modes[i], modes[j]))
                ys.append(label)
                ss.append(subj)
    if not xs:
        return np.empty((0, modes.shape[-1]), np.float32), np.empty(0, int), np.empty(0, str)
    return np.stack(xs).astype(np.float32), np.array(ys), np.array(ss)


def synthesize_all(modes: np.ndarray, y: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """modes (N, S, 8, T) over S subcarriers -> (x (M*S, T), y (M*S,))."""
    xs, ys = [], []
    for s in range(modes.shape[1]):
        sx, sy, _ = synthesize(modes[:, s], y, subject)
        xs.append(sx)
        ys.append(sy)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    from wisense.data import CACHE_DIR, TEST_SUBJECTS, load

    ap = argparse.ArgumentParser(description="EEMD-decompose the training recordings")
    ap.add_argument("--subcarriers", type=int, default=AUG_DEFAULT, help="how many of the 47 (spread evenly)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--trials", type=int, default=EEMD_TRIALS)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="clean",
                    help="clean: training subjects only. paper: all 22 subjects, as the "
                         "authors' scripts do -- test subjects leak into training")
    args = ap.parse_args()

    trn, tst = load().split()
    assert not np.isin(trn.subject, TEST_SUBJECTS).any()
    scs = spread_subcarriers(args.subcarriers)
    # Training-subject parts are shared by both protocols; the paper protocol
    # additionally decomposes the test subjects, into their own directory.
    sets = [("train", trn, CACHE_DIR / f"imfs_parts_t{args.trials}")]
    if args.protocol == "paper":
        sets.append(("test", tst, CACHE_DIR / f"imfs_parts_t{args.trials}_test"))

    all_modes, ys, subjects, paths = [], [], [], []
    for name, part_set, part_dir in sets:
        print(f"decomposing {len(part_set.y)} {name}-subject recordings x {len(scs)} subcarriers "
              f"(kept indices {scs}), {args.trials} trials each")
        # One file per subcarrier, written as soon as it finishes, so an
        # interrupted run resumes where it stopped instead of starting over.
        part_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for n, sc in enumerate(scs, start=1):
            part = part_dir / f"kept{sc:02d}.npy"
            if part.exists():
                print(f"[{name} {n}/{len(scs)}] kept {sc}: cached")
            else:
                print(f"[{name} {n}/{len(scs)}] kept {sc}: decomposing {len(part_set.y)} series", flush=True)
                np.save(part.with_suffix(".tmp.npy"), decompose_all(part_set.x[:, :, sc], args.workers, args.trials))
                part.with_suffix(".tmp.npy").replace(part)
            parts.append(np.load(part))
        all_modes.append(np.stack(parts, axis=1))          # (N, S, 8, T)
        ys.append(part_set.y)
        subjects.append(part_set.subject)
        paths.append(part_set.path)

    modes, y, subject = np.concatenate(all_modes), np.concatenate(ys), np.concatenate(subjects)
    out = CACHE_DIR / IMF_CACHE[args.protocol]
    np.savez(out, modes=modes, subcarriers=np.array(scs), y=y, subject=subject,
             path=np.concatenate(paths), protocol=args.protocol)
    sx, _ = synthesize_all(modes, y, subject)
    print(f"saved {out}; {len(sx):,} synthetic samples available ({args.protocol} protocol)")


if __name__ == "__main__":
    main()
