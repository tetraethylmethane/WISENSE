"""Is "ours beats the released model" statistically supported?

    python -m tools.stats

Both models are scored on the same 449 held-out recordings (paper protocol), so
the comparison is paired:

  * McNemar's exact test (two-sided binomial on the discordant recordings) for
    the two per-recording scorings, one subcarrier and vote.
  * A recording-level bootstrap (10,000 resamples) of the accuracy difference
    for all three scorings. The 47 per-subcarrier decisions of one recording are
    not independent, so per-subcarrier intervals resample recordings, never
    individual decisions.

Float and uint8 are tested separately. Writes artifacts/small_paper/stats.json.
"""

from __future__ import annotations

import json
from math import comb

import numpy as np

from wisense.csi import GOOD_KEPT
from wisense.data import RAW_DIR, ROOT, load
from wisense.evaluate import predict_tflite
from wisense.features import N_GOOD, per_subcarrier, top_variance_index

RUN = ROOT / "artifacts" / "small_paper"
RELEASED = RAW_DIR.parents[1] / "Embedded_Model"
N_BOOT = 10_000


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Two-sided exact McNemar test on paired per-recording correctness."""
    n01 = int(np.sum(~a_ok & b_ok))      # only B right
    n10 = int(np.sum(a_ok & ~b_ok))      # only A right
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_released_right": n10, "only_ours_right": n01, "p_value": p}


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """a, b: per-recording scores (correct fraction). CI of mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)],
            "p_le_0": float(np.mean(d <= 0))}


def per_recording(probs: np.ndarray, y: np.ndarray, top_pos: np.ndarray) -> dict:
    """probs (N, 47, 5) -> per-recording scores for the three scorings."""
    pred = probs.argmax(-1)
    vote = np.array([np.bincount(r, minlength=probs.shape[-1]).argmax() for r in pred])
    return {"per_subcarrier": (pred == y[:, None]).mean(1),
            "top_variance": (pred[np.arange(len(y)), top_pos] == y).astype(float),
            "vote": (vote == y).astype(float)}


def main() -> None:
    from tensorflow import keras

    _, tst = load().split()
    xs = per_subcarrier(tst.x)
    top_pos = np.array([GOOD_KEPT.index(k) for k in top_variance_index(tst.x)])
    shape = (len(tst.y), N_GOOD, -1)

    models = {
        ("released", "float"): lambda: keras.models.load_model(
            RELEASED / "small_model_complete_EMD_LSTM.keras", compile=False).predict(
            xs[..., None], batch_size=256, verbose=0),
        ("ours", "float"): lambda: keras.models.load_model(
            RUN / "model.keras", compile=False).predict(xs[..., None], batch_size=256, verbose=0),
        ("released", "uint8"): lambda: predict_tflite(
            (RELEASED / "small_model_complete_EMD_LSTM_uint8.tflite").read_bytes(), xs),
        ("ours", "uint8"): lambda: predict_tflite((RUN / "model_uint8.tflite").read_bytes(), xs),
    }
    scores = {}
    for key, fn in models.items():
        print(f"scoring {key[0]} {key[1]} ...", flush=True)
        scores[key] = per_recording(np.asarray(fn(), dtype=np.float32).reshape(shape), tst.y, top_pos)

    out = {"n_recordings": int(len(tst.y)), "bootstrap_resamples": N_BOOT, "protocol": "paper"}
    for prec in ("float", "uint8"):
        a, b = scores[("released", prec)], scores[("ours", prec)]
        out[prec] = {}
        print(f"\n{prec}: ours - released  (95% CI, recording bootstrap; McNemar exact p)")
        for m in ("per_subcarrier", "top_variance", "vote"):
            r = {"released": float(a[m].mean()), "ours": float(b[m].mean()), **bootstrap_diff(a[m], b[m])}
            if m != "per_subcarrier":
                r["mcnemar"] = mcnemar_exact(a[m] > 0.5, b[m] > 0.5)
            out[prec][m] = r
            mc = f"  McNemar p={r['mcnemar']['p_value']:.3g} " \
                 f"(only ours right {r['mcnemar']['only_ours_right']}, " \
                 f"only released right {r['mcnemar']['only_released_right']})" if "mcnemar" in r else ""
            print(f"  {m:15s} {r['released']:.4f} -> {r['ours']:.4f}  diff {r['diff']:+.4f} "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]{mc}")
    (RUN / "stats.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {RUN / 'stats.json'}")


if __name__ == "__main__":
    main()
