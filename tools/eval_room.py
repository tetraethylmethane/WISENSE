"""Cross-environment test: the paper-protocol model in a room it has never seen.

Recordings come from host/csi_capture.py record, in the dataset's own layout:
    <root>/<ACT>/<subject>/csi_data_*.txt     ACT in FA GU LD SD WA EM
EM ("empty room") is not in the paper's dataset; it is only used when
fine-tuning, as a sixth class.

Three experiments, all scored with the same three scorings as evaluate.py:

  zero-shot  the trained model unchanged, on the five paper activities.
  finetune   the trained model with a new 6-class head (EM added), trained
             briefly on some repetitions and tested on the rest.
  scratch    the same architecture and schedule from random initialisation --
             the baseline that says whether the paper-trained weights help.

finetune/scratch use stratified k-fold cross-validation over repetitions, so
every recording is tested exactly once by a model that never saw it. With one
subject and one session the folds are not independent in time; read the
result as "adaptation to this room", not as generalisation to new people.

    python -m tools.eval_room path/to/room_recordings
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from wisense.csi import GOOD_KEPT, TIMESTEPS, preprocess, read_raw
from wisense.data import ACTIVITIES, ROOT
from wisense.evaluate import metrics, predict_tflite
from wisense.features import N_GOOD, per_subcarrier, top_variance_index

ROOM_LABELS = ACTIVITIES + ("EM",)
RUN = ROOT / "artifacts" / "small_paper"


def load_room(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """(N, 850, 52) amplitude, (N,) label index into ROOM_LABELS, paths, skipped."""
    xs, ys, paths, skipped = [], [], [], []
    for li, act in enumerate(ROOM_LABELS):
        for f in sorted((root / act).glob("*/*.txt")) if (root / act).exists() else []:
            _, csi = read_raw(f)
            if len(csi) < TIMESTEPS:
                skipped.append(f"{f.relative_to(root)} ({len(csi)} packets)")
                continue
            xs.append(preprocess(csi).astype(np.float32))
            ys.append(li)
            paths.append(str(f.relative_to(root)))
    return np.stack(xs), np.array(ys), paths, skipped


def normalise_packets(x: np.ndarray) -> np.ndarray:
    """Divide each packet's amplitudes by their mean over the 47 good subcarriers.

    Cancels the receiver's automatic gain changes, which scale every subcarrier
    of a packet together. Not part of the paper's preprocessing; an option for
    the cross-environment experiment only.
    """
    g = x[..., list(GOOD_KEPT)].mean(axis=-1, keepdims=True)
    return (x / (g + 1e-6)).astype(np.float32)


def score(probs: np.ndarray, y: np.ndarray, top_pos: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(-1)
    vote = np.array([np.bincount(r, minlength=n_classes).argmax() for r in pred])
    m = lambda yt, yp: metrics_n(yt, yp, n_classes)
    return {"per_subcarrier": m(np.repeat(y, N_GOOD), pred.ravel()),
            "top_variance": m(y, pred[np.arange(len(y)), top_pos]),
            "vote": m(y, vote)}


def metrics_n(y: np.ndarray, p: np.ndarray, n: int) -> dict:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    return {"n": int(len(y)), "accuracy": float(tp.sum() / cm.sum()),
            "macro_recall": float(rec[cm.sum(1) > 0].mean()),
            "recall": dict(zip(ROOM_LABELS[:n], rec.round(3).tolist())), "confusion": cm.tolist()}


def zero_shot(x, y) -> dict:
    from tensorflow import keras

    keep = y < len(ACTIVITIES)               # the model knows only the paper's five
    xa, ya = x[keep], y[keep]
    top_pos = np.array([GOOD_KEPT.index(k) for k in top_variance_index(xa)])
    xs = per_subcarrier(xa)[..., None]
    model = keras.models.load_model(RUN / "model.keras")
    res = {"float": score(model.predict(xs, batch_size=256, verbose=0).reshape(len(ya), N_GOOD, -1),
                          ya, top_pos, len(ACTIVITIES))}
    tfl = RUN / "model_uint8.tflite"
    if tfl.exists():
        res["uint8"] = score(predict_tflite(tfl.read_bytes(), xs[..., 0]).reshape(len(ya), N_GOOD, -1),
                             ya, top_pos, len(ACTIVITIES))
    em = x[y == ROOM_LABELS.index("EM")]
    if len(em):
        p = model.predict(per_subcarrier(em)[..., None], batch_size=256, verbose=0).argmax(-1)
        res["empty_room_predicted_as"] = {a: int((p == i).sum()) for i, a in enumerate(ACTIVITIES)}
    return res


def build_six_class(pretrained: bool):
    from tensorflow import keras

    from wisense.models import PAPER_SMALL, build, compile_paper
    if pretrained:
        base = keras.models.load_model(RUN / "model.keras")
    else:
        base = build(PAPER_SMALL)
    feat = base.get_layer("global_average_pooling1d").output
    out = keras.layers.Dense(len(ROOM_LABELS), activation="softmax", name="dense_room")(feat)
    return compile_paper(keras.Model(base.input, out))


def room_modes(x: np.ndarray, paths: list[str], workers: int | None = None) -> np.ndarray:
    """EEMD modes of every good subcarrier of every room recording, cached.

    (N, 47, 8, 850). Same parameters as the paper's augmentation (augment.py).
    """
    from wisense.augment import N_MODES, decompose_all
    from wisense.data import CACHE_DIR

    cache = CACHE_DIR / "room_imfs.npz"
    if cache.exists():
        z = np.load(cache)
        if list(z["paths"]) == paths:
            return z["modes"]
    series = x[:, :, list(GOOD_KEPT)].transpose(0, 2, 1).reshape(-1, x.shape[1])
    print(f"EEMD of {len(series)} room series (paper parameters) ...", flush=True)
    modes = decompose_all(series, workers).reshape(len(x), N_GOOD, N_MODES, -1)
    np.savez(cache, modes=modes, paths=np.array(paths))
    return modes


def cross_validate(x, y, pretrained: bool, folds: int, epochs: int, seed: int,
                   modes: np.ndarray | None = None) -> dict:
    """modes given: add EEMD-synthetic samples built from each fold's TRAINING
    recordings only, so no test repetition contributes to training (cf. F1)."""
    from tensorflow import keras

    from wisense.augment import synthesize_all

    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):                    # stratified: each class spread over folds
        idx = rng.permutation(np.flatnonzero(y == c))
        fold_of[idx] = np.arange(len(idx)) % folds
    probs = np.zeros((len(y), N_GOOD, len(ROOM_LABELS)), dtype=np.float32)
    for k in range(folds):
        keras.utils.set_random_seed(seed + k)
        tr, te = fold_of != k, fold_of == k
        model = build_six_class(pretrained)
        xtr, ytr_idx = per_subcarrier(x[tr]), np.repeat(y[tr], N_GOOD)
        if modes is not None:
            sx, sy = synthesize_all(modes[tr], y[tr], np.array(["r1"] * int(tr.sum())))
            xtr, ytr_idx = np.concatenate([xtr, sx]), np.concatenate([ytr_idx, sy])
            if k == 0:
                print(f"  fold training set: {int(tr.sum()) * N_GOOD} real + {len(sy)} synthetic samples")
        xtr = xtr[..., None]
        ytr = keras.utils.to_categorical(ytr_idx, len(ROOM_LABELS))
        model.fit(xtr, ytr, batch_size=32, epochs=epochs, shuffle=True, verbose=0)
        probs[te] = model.predict(per_subcarrier(x[te])[..., None], batch_size=256,
                                  verbose=0).reshape(int(te.sum()), N_GOOD, -1)
        print(f"  fold {k + 1}/{folds} done", flush=True)
    top_pos = np.array([GOOD_KEPT.index(k) for k in top_variance_index(x)])
    return score(probs, y, top_pos, len(ROOM_LABELS))


def fit_check(x, y, pretrained: bool, epochs: int, seed: int) -> dict:
    """Train on every recording and score on the same ones.

    Not a generalisation measure -- a diagnostic. If the model cannot fit the
    recordings it trains on, poor cross-validation reflects the training
    schedule or the data, not the room.
    """
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    model = build_six_class(pretrained)
    xs = per_subcarrier(x)[..., None]
    hist = model.fit(xs, keras.utils.to_categorical(np.repeat(y, N_GOOD), len(ROOM_LABELS)),
                     batch_size=32, epochs=epochs, shuffle=True, verbose=0)
    probs = model.predict(xs, batch_size=256, verbose=0).reshape(len(y), N_GOOD, -1)
    top_pos = np.array([GOOD_KEPT.index(k) for k in top_variance_index(x)])
    res = score(probs, y, top_pos, len(ROOM_LABELS))
    res["train_acc_by_epoch"] = [round(float(a), 4) for a in hist.history["accuracy"]]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=("zero-shot", "finetune", "scratch", "fitcheck",
                                       "finetune-eemd"), default=None)
    ap.add_argument("--workers", type=int, default=None, help="EEMD processes")
    ap.add_argument("--normalise", action="store_true",
                    help="per-packet amplitude normalisation (cancels AGC gain jumps)")
    args = ap.parse_args()

    x, y, paths, skipped = load_room(args.root)
    print(f"{len(y)} recordings: " + " ".join(f"{a}={int((y == i).sum())}" for i, a in enumerate(ROOM_LABELS)))
    for s in skipped:
        print(f"  skipped (shorter than {TIMESTEPS} packets): {s}")

    if args.normalise:
        x = normalise_packets(x)
        print("per-packet normalisation ON")
    out = {"recordings": len(y), "skipped": skipped, "folds": args.folds, "epochs": args.epochs,
           "normalised": args.normalise}
    if args.only in (None, "zero-shot"):
        out["zero_shot"] = r = zero_shot(x, y)
        print("\nZERO-SHOT (paper model, 5 activities):")
        for kind in ("float", "uint8"):
            if kind in r:
                print(f"  {kind:5s} " + "  ".join(f"{m} {v['accuracy']:.3f}" for m, v in r[kind].items()))
        if "empty_room_predicted_as" in r:
            print(f"  empty room predicted as: {r['empty_room_predicted_as']}")
    for name, pre in (("finetune", True), ("scratch", False)):
        if args.only in (None, name):
            print(f"\n{name.upper()} (6 classes incl. EM, {args.folds}-fold CV, {args.epochs} epochs):")
            out[name] = r = cross_validate(x, y, pre, args.folds, args.epochs, args.seed)
            print("  float " + "  ".join(f"{m} {v['accuracy']:.3f}" for m, v in r.items()))
            print("  recall (vote): " + " ".join(f"{a}:{v:.2f}" for a, v in r["vote"]["recall"].items()))
    if args.only == "finetune-eemd":
        modes = room_modes(x, paths, args.workers)
        print(f"\nFINETUNE + EEMD (6 classes, {args.folds}-fold CV, {args.epochs} epochs, "
              f"synthetic from training folds only):")
        out["finetune_eemd"] = r = cross_validate(x, y, True, args.folds, args.epochs, args.seed, modes)
        print("  float " + "  ".join(f"{m} {v['accuracy']:.3f}" for m, v in r.items()))
        print("  recall (vote): " + " ".join(f"{a}:{v:.2f}" for a, v in r["vote"]["recall"].items()))
    if args.only == "fitcheck":
        for name, pre in (("fitcheck_pretrained", True), ("fitcheck_scratch", False)):
            print(f"\n{name.upper()} (train and score on all {len(y)} recordings, {args.epochs} epochs):")
            out[name] = r = fit_check(x, y, pre, args.epochs, args.seed)
            curve = r["train_acc_by_epoch"]
            print("  train acc by epoch: " + " ".join(f"{a:.2f}" for a in curve[::max(1, len(curve) // 10)]))
            print("  scored on same data: " + "  ".join(f"{m} {r[m]['accuracy']:.3f}"
                                                    for m in ("per_subcarrier", "top_variance", "vote")))
    suffix = "_norm" if args.normalise else ""
    dest = RUN / {"fitcheck": "room_fitcheck.json",
                  "finetune-eemd": "room_eemd.json"}.get(args.only, "room_eval.json").replace(".json", suffix + ".json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
