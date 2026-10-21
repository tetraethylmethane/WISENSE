"""Train the LSTM-DenseNet on real + EEMD-synthetic training-subject samples.

Hyperparameters are the paper's: Adam 1e-4, batch 32, no validation-based
stopping (the paper reports none). The authors did not release a training
script; this is reconstructed from the paper and from the compile_config
saved inside their .keras files.

Every good subcarrier of every recording is its own (850, 1) sample -- see
features.py for how that was established. That makes the pool large:
1,198 recordings x 47 = 56k real samples, plus the synthetic ones. The
paper trained 300 full passes over it on a GPU, and the defaults here are the
paper's: --epochs 300, --per-epoch 0 (full pass), Adam 1e-4, batch 32. That
needs a GPU (~half a day); on a laptop CPU it is over a week. --per-epoch N
draws N random samples per epoch instead, for smoke tests only.

The test subjects are never loaded here. Scoring is evaluate.py's job.

    python -m wisense.augment                 # once: EEMD of train recordings
    python -m wisense.train --arch small      # the paper's deployed model
    python -m wisense.train --arch large      # LSTM-DenseNet121 baseline
    python -m wisense.train --epochs 2 --per-epoch 640 --out artifacts/smoke   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wisense.augment import IMF_CACHE, PROTOCOLS, synthesize_all
from wisense.data import ACTIVITIES, CACHE_DIR, ROOT, TEST_SUBJECTS, load
from wisense.features import N_GOOD, per_subcarrier
from wisense.models import ARCHS, build, compile_paper

ARTIFACTS = ROOT / "artifacts"


def training_pool(use_real: bool = True, use_synthetic: bool = True,
                  protocol: str = "clean") -> tuple[np.ndarray, np.ndarray, dict]:
    """(N, 850) float32, (N,) int, counts.

    Real samples are always training subjects only, as in the paper. Under the
    clean protocol so are the synthetic ones; under the paper protocol the
    synthetic set is built from all 22 subjects, as the authors' scripts did.
    """
    trn, _ = load().split()
    xs, ys, counts = [], [], {}
    if use_real:
        xs.append(per_subcarrier(trn.x))
        ys.append(np.repeat(trn.y, N_GOOD))
        counts["real"] = len(ys[-1])
    if use_synthetic:
        cache = CACHE_DIR / IMF_CACHE[protocol]
        if not cache.exists():
            raise FileNotFoundError(f"{cache} missing -- run "
                                    f"`python -m wisense.augment --protocol {protocol}` first")
        z = np.load(cache)
        leaked = np.isin(z["subject"], TEST_SUBJECTS).any()
        if protocol == "clean" and leaked:
            raise AssertionError("clean IMF cache contains test subjects; rebuild it")
        if protocol == "paper" and not leaked:
            raise AssertionError("paper-protocol cache lacks test subjects; rebuild it")
        sx, sy = synthesize_all(z["modes"], z["y"], z["subject"])
        xs.append(sx)
        ys.append(sy)
        counts["synthetic"] = len(sy)
        counts["synthetic_subcarriers"] = int(z["modes"].shape[1])
        counts["synthetic_includes_test_subjects"] = bool(leaked)
    return np.concatenate(xs), np.concatenate(ys), counts


def make_sampler(x: np.ndarray, y: np.ndarray, per_epoch: int, batch: int, seed: int):
    from tensorflow import keras

    class RandomEpochs(keras.utils.PyDataset):
        """Each epoch a shuffled pass over the pool, or a random subset of it."""

        def __init__(self):
            super().__init__()
            self.n = per_epoch if per_epoch > 0 else len(x)
            self.rng = np.random.default_rng(seed)
            self.on_epoch_end()

        def __len__(self):
            return self.n // batch

        def __getitem__(self, i):
            idx = np.sort(self.order[i * batch:(i + 1) * batch])
            return x[idx, :, None], keras.utils.to_categorical(y[idx], len(ACTIVITIES))

        def on_epoch_end(self):
            self.order = self.rng.choice(len(x), min(self.n, len(x)), replace=False)

    return RandomEpochs()


def main() -> None:
    import tensorflow as tf
    from tensorflow import keras

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=ARCHS, default="small")
    ap.add_argument("--epochs", type=int, default=300, help="paper: 300")
    ap.add_argument("--per-epoch", type=int, default=0,
                    help="samples per epoch; 0 = a full pass over the pool, as the paper trains")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Adam learning rate; the paper's 1e-4 assumes its GPU-scale budget")
    ap.add_argument("--plateau", action="store_true",
                    help="halve the LR when training loss stalls for 8 epochs (floor 1e-5)")
    ap.add_argument("--no-synthetic", action="store_true", help="real samples only (the paper's ablation)")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="clean",
                    help="paper = synthetic samples from all subjects, test included, as the "
                         "authors did; clean = training subjects only")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after the epoch that crosses this wall-clock limit "
                         "(Kaggle/Colab sessions are killed at ~12 h); --resume continues")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the run directory's last saved epoch")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    x, y, counts = training_pool(use_synthetic=not args.no_synthetic, protocol=args.protocol)
    print(f"pool {len(y):,} samples {counts}; "
          + " ".join(f"{a}={int(np.sum(y == i))}" for i, a in enumerate(ACTIVITIES)))

    suffix = "_real" if args.no_synthetic else ("_paper" if args.protocol == "paper" else "")
    out = args.out or ARTIFACTS / f"{args.arch}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    # The checkpoint is written after every epoch, so a run that is
    # interrupted loses at most one epoch. --resume picks it back up.
    initial_epoch = 0
    if args.resume and (out / "model.keras").exists() and (out / "history.csv").exists():
        model = keras.models.load_model(out / "model.keras")
        # Count rows, not line breaks: on Windows CSVLogger's file has "\r\r\n"
        # endings, which splitlines() reads as two lines per row.
        rows = [r for r in (out / "history.csv").read_text().splitlines() if r.strip()]
        initial_epoch = int(rows[-1].split(",")[0]) + 1 if len(rows) > 1 else 0
        model.optimizer.learning_rate.assign(args.lr)
        print(f"resuming {out} at epoch {initial_epoch}, lr {args.lr:g}")
    else:
        model = compile_paper(build(ARCHS[args.arch]), args.lr)
    print(f"{args.arch}: {model.count_params():,} parameters; "
          f"{args.epochs} epochs x {args.per_epoch:,} samples")

    callbacks = [keras.callbacks.CSVLogger(str(out / "history.csv"), append=initial_epoch > 0),
                 keras.callbacks.ModelCheckpoint(str(out / "model.keras"))]
    if args.max_hours:
        deadline = time.time() + args.max_hours * 3600

        class StopAtDeadline(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if time.time() > deadline:
                    print(f"--max-hours reached after epoch {epoch + 1}; checkpoint saved, "
                          f"rerun with --resume to continue", flush=True)
                    self.model.stop_training = True

        callbacks.append(StopAtDeadline())
    if args.plateau:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1))

    t0 = time.time()
    hist = model.fit(make_sampler(x, y, args.per_epoch, args.batch, args.seed + initial_epoch),
                     epochs=args.epochs, initial_epoch=initial_epoch, verbose=2,
                     callbacks=callbacks)
    model.save(out / "model.keras")
    (out / "train.json").write_text(json.dumps({
        "arch": args.arch, "epochs": args.epochs, "per_epoch": args.per_epoch,
        "batch": args.batch, "lr": args.lr, "plateau": args.plateau, "seed": args.seed,
        "protocol": args.protocol,
        "resumed_from_epoch": initial_epoch, "pool": counts,
        "params": int(model.count_params()),
        "final_train_acc": float(hist.history["accuracy"][-1]) if hist.history else None,
        "epochs_done": initial_epoch + len(hist.history.get("accuracy", [])),
        "seconds": round(time.time() - t0), "tensorflow": tf.__version__,
    }, indent=2))
    print(f"saved {out / 'model.keras'}")


if __name__ == "__main__":
    main()
