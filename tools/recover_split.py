"""Recover which subjects the paper held out, from the released x_trn.npy.

x_trn.npy is the authors' real training set: (1198, 52, 850), each sample the
sum of its EEMD modes -- approximately, not exactly, the amplitude (EEMD's
added noise does not cancel perfectly). So match by correlation over the full
recording rather than by equality, and report how unambiguous each match is.

    python -m tools.recover_split
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from wisense.data import ACTIVITIES, RAW_DIR, SUBJECTS, load


def _zscore(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ds = load()
    released = np.load(RAW_DIR.parent / "x_trn.npy", mmap_mode="r")
    released_y = np.load(RAW_DIR.parent / "y_trn.npy").astype(int) - 1

    # (N, 52*850) in the same (subcarrier, time) layout as the release.
    ours = _zscore(ds.x.transpose(0, 2, 1).reshape(len(ds.y), -1).astype(np.float32))
    best = np.empty(len(released), dtype=int)
    margin = np.empty(len(released))
    for i0 in range(0, len(released), 128):
        chunk = np.asarray(released[i0:i0 + 128], dtype=np.float32).reshape(-1, ours.shape[1])
        corr = _zscore(chunk) @ ours.T
        top2 = np.sort(corr, axis=1)[:, -2:]
        best[i0:i0 + 128] = corr.argmax(axis=1)
        margin[i0:i0 + 128] = top2[:, 1] - top2[:, 0]

    print(f"matched {len(released)} released samples; "
          f"{len(set(best.tolist()))} distinct recordings; "
          f"label agreement {np.mean(ds.y[best] == released_y):.4f}; "
          f"median top-1 margin {np.median(margin):.3f}")
    present = Counter(ds.subject[best].tolist())
    absent = [s for s in SUBJECTS if s not in present]
    for s in SUBJECTS:
        print(f"  {s:4s} {present.get(s, 0):4d} matches  of {int(np.sum(ds.subject == s))}")
    print("held out (absent from x_trn):", " ".join(absent))
    print("labels:", " ".join(ACTIVITIES))


if __name__ == "__main__":
    main()
