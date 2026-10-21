"""Score a trained run on the six held-out subjects, float and uint8.

Three scorings of the same predictions (see features.py):
  per_subcarrier  the paper's headline metric
  top_variance    what the device does: one subcarrier, one inference
  vote            all 47 subcarriers, majority vote

The paper's numbers were obtained with test-subject recordings recombined
into the training set (see data.py), so landing below them is expected and is
the honest figure. Landing *above* them deserves suspicion.

    python -m wisense.evaluate artifacts/small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wisense.csi import GOOD_KEPT
from wisense.data import ACTIVITIES, load
from wisense.features import N_GOOD, per_subcarrier, top_variance_index

# Paper Table values, single-subcarrier configuration (= our per_subcarrier).
PAPER = {
    "small": {"float": 0.9733, "uint8": 0.9243, "latency_ms": 232, "size_kb": 127},
    "large": {"float": 0.9755, "uint8": 0.9555, "latency_ms": 817, "size_kb": 277},
}


def confusion(y: np.ndarray, p: np.ndarray, n: int = len(ACTIVITIES)) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (y, p), 1)
    return m


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = confusion(y, p)
    tp = np.diag(m).astype(float)
    recall = tp / np.maximum(m.sum(1), 1)
    precision = tp / np.maximum(m.sum(0), 1)
    return {"n": int(len(y)), "accuracy": float(tp.sum() / m.sum()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "precision": dict(zip(ACTIVITIES, precision.round(4).tolist())),
            "recall": dict(zip(ACTIVITIES, recall.round(4).tolist())),
            "confusion": m.tolist()}


def score_all(probs: np.ndarray, y: np.ndarray, top_pos: np.ndarray) -> dict:
    """probs (N, 47, 5) -> metrics for the three scorings."""
    n = len(y)
    pred = probs.argmax(-1)
    vote = np.array([np.bincount(r, minlength=len(ACTIVITIES)).argmax() for r in pred])
    return {"per_subcarrier": metrics(np.repeat(y, N_GOOD), pred.ravel()),
            "top_variance": metrics(y, pred[np.arange(n), top_pos]),
            "vote": metrics(y, vote)}


def predict_tflite(tflite: bytes, x: np.ndarray) -> np.ndarray:
    """(N, 850) float -> (N, 5) uint8 scores."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zero = i["quantization"]
    out = np.empty((len(x), len(ACTIVITIES)), dtype=np.uint8)
    for k, s in enumerate(x):
        q = np.clip(np.round(s / scale + zero), 0, 255).astype(np.uint8)
        interp.set_tensor(i["index"], q[None, :, None])
        interp.invoke()
        out[k] = interp.get_tensor(o["index"])[0]
    return out


def report(name: str, r: dict) -> str:
    m = np.array(r["confusion"])
    lines = [f"{name}: accuracy {r['accuracy']:.4f}  macro-P {r['macro_precision']:.4f}  "
             f"macro-R {r['macro_recall']:.4f}  (n={r['n']})",
             "        " + " ".join(f"{a:>6s}" for a in ACTIVITIES) + "   recall"]
    for a, row, rec in zip(ACTIVITIES, m, r["recall"].values()):
        lines.append(f"  {a:4s}  " + " ".join(f"{v:6d}" for v in row) + f"   {rec:.3f}")
    return "\n".join(lines)


def evaluate_models(keras_model, tflite: bytes | None) -> dict:
    _, tst = load().split()
    xs = per_subcarrier(tst.x)
    top_pos = np.array([GOOD_KEPT.index(k) for k in top_variance_index(tst.x)])
    res = {"float": score_all(keras_model.predict(xs[..., None], batch_size=256, verbose=0)
                              .reshape(len(tst.y), N_GOOD, -1), tst.y, top_pos)}
    if tflite is not None:
        res["uint8"] = score_all(predict_tflite(tflite, xs).reshape(len(tst.y), N_GOOD, -1), tst.y, top_pos)
        res["uint8_bytes"] = len(tflite)
    return res


def main() -> None:
    from tensorflow import keras

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    args = ap.parse_args()

    meta = json.loads((args.run / "train.json").read_text())
    arch, protocol = meta["arch"], meta.get("protocol", "clean")
    tfl = args.run / "model_uint8.tflite"
    res = evaluate_models(keras.models.load_model(args.run / "model.keras"),
                          tfl.read_bytes() if tfl.exists() else None)

    print(f"held-out subjects s4-s9, protocol={protocol}")
    if protocol == "paper":
        print("NOTE: paper protocol -- synthetic training samples were built from these test "
              "subjects' recordings, exactly as the authors did. Not a clean held-out score.")
    print()
    for k in ("float", "uint8"):
        if k not in res:
            continue
        print(f"===== {k}  (paper, per-subcarrier: {PAPER[arch][k]:.4f}) =====")
        for mode, r in res[k].items():
            print(report(mode, r) + "\n")
    if "uint8" in res:
        print("summary  " + "  ".join(
            f"{mode}: float {res['float'][mode]['accuracy']:.4f} -> uint8 {res['uint8'][mode]['accuracy']:.4f}"
            for mode in res["float"]))
        print(f"model file {res['uint8_bytes']:,} B (paper reports {PAPER[arch]['size_kb']} kB)")
    res["protocol"] = protocol
    (args.run / "evaluation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
