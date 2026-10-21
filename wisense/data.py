"""The paper's activity dataset, and the subject-held-out split.

Dataset: 22 volunteers x 5 activities x ~15 repetitions of 20 s, collected
with two ESP32s at 50 packets/s. Raw files live in the authors' repository,
cloned next to this one (or one level further up):

    Embedded_WiFi_Sensing/Datasets/Activity/<ACT>/<subject>/csi_data_*.txt

Override with WISENSE_RAW=<path to the Activity directory>.

THE SPLIT, AND THE LEAK IN THE ORIGINAL
---------------------------------------
The authors hold out whole subjects for testing, which is the right protocol.
But their imf_sample_builder.py builds synthetic samples from *every*
subject, and dataset_constructor.py puts all of them in the training set --
so recombined test-subject recordings were trained on. The arithmetic gives
it away: 22 subjects x 5 activities x C(15,2) pairs = 11,550, against the
paper's 11,513 synthetic samples. Held-out subjects alone would give ~8,400.

Here the default (clean) protocol augments TRAIN_SUBJECTS only; the authors'
behaviour is available, explicitly labelled, as --protocol paper so their
numbers can be reproduced as obtained. assert_no_subject_overlap() runs on
import so the split itself cannot regress.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wisense.csi import N_KEPT, TIMESTEPS, preprocess, read_raw

ROOT = Path(__file__).resolve().parents[1]


def _default_raw_dir() -> Path:
    for base in (ROOT.parent, ROOT.parents[1]):     # sibling clone, or one level up
        cand = base / "Embedded_WiFi_Sensing" / "Datasets" / "Activity"
        if cand.is_dir():
            return cand
    return ROOT.parent / "Embedded_WiFi_Sensing" / "Datasets" / "Activity"


RAW_DIR = Path(os.environ.get("WISENSE_RAW", _default_raw_dir()))
CACHE_DIR = Path(os.environ.get("WISENSE_CACHE", ROOT / "data"))

# The authors' label order (dataset_constructor.py: 1-FA ... 5-WA), zero-based.
ACTIVITIES = ("FA", "GU", "LD", "SD", "WA")
ACTIVITY_NAMES = {"FA": "Fall", "GU": "Get up", "LD": "Lie down",
                  "SD": "Sit down", "WA": "Walk"}

SUBJECTS = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
            "s12", "s13", "s14", "s15", "s16", "s17", "s19", "s21", "s22",
            "s23", "s24")  # s18 and s20 do not exist in the release

# --------------------------------------------------------------------------
# Held-out subjects.
#
# dataset_constructor.py takes "the last 6" of os.listdir(), whose order is
# filesystem-dependent. Which six the paper used was recovered by matching the
# released x_trn.npy against the raw recordings (tools/recover_split.py):
# these are the subjects absent from it.
# --------------------------------------------------------------------------
TEST_SUBJECTS = ("s4", "s5", "s6", "s7", "s8", "s9")
TRAIN_SUBJECTS = tuple(s for s in SUBJECTS if s not in TEST_SUBJECTS)


def assert_no_subject_overlap() -> None:
    overlap = set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS)
    assert not overlap, f"subjects in both train and test: {sorted(overlap)}"
    assert set(TRAIN_SUBJECTS) | set(TEST_SUBJECTS) == set(SUBJECTS)


assert_no_subject_overlap()


@dataclass(frozen=True)
class Recording:
    activity: str
    subject: str
    path: Path

    @property
    def label(self) -> int:
        return ACTIVITIES.index(self.activity)


def index(raw_dir: Path = RAW_DIR) -> list[Recording]:
    """Every recording, in a fixed order (activity, subject number, filename)."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_dir} not found. Clone the authors' repo next to Alpha/ with "
            "git-lfs installed, or set WISENSE_RAW.")
    recs = []
    for act in ACTIVITIES:
        for subj in sorted(os.listdir(raw_dir / act), key=lambda s: int(s[1:])):
            for f in sorted(os.listdir(raw_dir / act / subj)):
                recs.append(Recording(act, subj, raw_dir / act / subj / f))
    return recs


@dataclass
class Dataset:
    x: np.ndarray          # (N, 850, 52) float32 amplitude
    y: np.ndarray          # (N,) int64 label
    subject: np.ndarray    # (N,) str
    path: np.ndarray       # (N,) str, relative to RAW_DIR

    def where(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.x[mask], self.y[mask], self.subject[mask], self.path[mask])

    def split(self) -> tuple["Dataset", "Dataset"]:
        test = np.isin(self.subject, TEST_SUBJECTS)
        return self.where(~test), self.where(test)


def load(raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR) -> Dataset:
    """All recordings as amplitude, cached after the first (slow) parse."""
    cache = cache_dir / "amplitude.npz"
    if cache.exists():
        z = np.load(cache)
        return Dataset(z["x"], z["y"], z["subject"], z["path"])

    recs = index(raw_dir)
    x = np.empty((len(recs), TIMESTEPS, N_KEPT), dtype=np.float32)
    for i, r in enumerate(recs):
        _, csi = read_raw(r.path)
        x[i] = preprocess(csi)
        if i % 100 == 0:
            print(f"  parsed {i}/{len(recs)}", flush=True)
    ds = Dataset(x, np.array([r.label for r in recs]),
                 np.array([r.subject for r in recs]),
                 np.array([r.path.relative_to(raw_dir).as_posix() for r in recs]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=ds.x, y=ds.y, subject=ds.subject, path=ds.path)
    return ds


def describe(ds: Dataset) -> str:
    trn, tst = ds.split()
    lines = [f"{len(ds)} recordings, x {ds.x.shape}",
             f"train subjects ({len(TRAIN_SUBJECTS)}): {' '.join(TRAIN_SUBJECTS)}",
             f"test subjects  ({len(TEST_SUBJECTS)}): {' '.join(TEST_SUBJECTS)}",
             "            " + " ".join(f"{a:>5s}" for a in ACTIVITIES)]
    for name, part in (("train", trn), ("test", tst)):
        c = Counter(part.y.tolist())
        lines.append(f"  {name:8s}  " + " ".join(f"{c[i]:5d}" for i in range(len(ACTIVITIES)))
                     + f"   = {len(part)}")
    return "\n".join(lines)


Dataset.__len__ = lambda self: len(self.y)  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="build the cache and print the split")
    ap.parse_args()
    print(describe(load()))


if __name__ == "__main__":
    main()
