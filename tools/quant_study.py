"""Does a larger calibration set reduce the 8-bit accuracy loss?

    python -m tools.quant_study

Post-training full-integer quantisation (wisense.export.quantise) is
calibrated on N randomly drawn training-subject per-subcarrier samples. The
paper does not state N; our export uses 300. This re-quantises the trained
paper-protocol model with several N and scores each on the held-out subjects
with the same three scorings as evaluate.py. Weights are unchanged -- only the
activation ranges the converter picks depend on N.

Writes artifacts/small_paper/quant_study.json.
"""

from __future__ import annotations

import json

import numpy as np

from wisense.data import ROOT, load
from wisense.evaluate import predict_tflite
from wisense.export import quantise
from wisense.features import N_GOOD, per_subcarrier, top_variance_index
from wisense.csi import GOOD_KEPT

RUN = ROOT / "artifacts" / "small_paper"
SIZES = (300, 1000, 3000)


def main() -> None:
    from tensorflow import keras

    trn, tst = load().split()
    pool = per_subcarrier(trn.x)
    xs = per_subcarrier(tst.x)
    top_pos = np.array([GOOD_KEPT.index(k) for k in top_variance_index(tst.x)])
    model = keras.models.load_model(RUN / "model.keras")

    float_p = model.predict(xs[..., None], batch_size=256, verbose=0).reshape(len(tst.y), N_GOOD, -1)
    res = {"float": score(float_p, tst.y, top_pos)}
    print("float  " + fmt(res["float"]), flush=True)
    for n in SIZES:
        rng = np.random.default_rng(0)
        calib = pool[rng.choice(len(pool), n, replace=False)][..., None]
        tfl = quantise(model, calib)
        p = predict_tflite(tfl, xs).astype(np.float32).reshape(len(tst.y), N_GOOD, -1)
        res[f"uint8_calib{n}"] = r = score(p, tst.y, top_pos)
        r["bytes"] = len(tfl)
        print(f"calib {n:5d}  " + fmt(r), flush=True)
    (RUN / "quant_study.json").write_text(json.dumps(res, indent=2))
    print(f"saved {RUN / 'quant_study.json'}")


def score(probs: np.ndarray, y: np.ndarray, top_pos: np.ndarray) -> dict:
    pred = probs.argmax(-1)
    vote = np.array([np.bincount(r, minlength=probs.shape[-1]).argmax() for r in pred])
    return {"per_subcarrier": float((pred == y[:, None]).mean()),
            "top_variance": float((pred[np.arange(len(y)), top_pos] == y).mean()),
            "vote": float((vote == y).mean())}


def fmt(r: dict) -> str:
    return "  ".join(f"{k} {r[k]:.4f}" for k in ("per_subcarrier", "top_variance", "vote"))


if __name__ == "__main__":
    main()
