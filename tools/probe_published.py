"""Reverse-engineer the published models' input, and score them as released.

The authors released trained models but no training or inference code, and
the paper does not say how a "single subcarrier" input is chosen. This runs
their small model over every one of the 52 subcarriers of every held-out
recording, which shows:

  * it was trained per subcarrier -- any one good subcarrier scores ~0.9+
  * five subcarriers (raw 1, 38-41) are useless -- the paper's unnamed
    "problematic" ones, 52 - 5 = 47
  * its per-subcarrier accuracy over the 47 is the paper's 92.43%

Caveat on every number printed: these models were trained on recombined
recordings of the test subjects (wisense/data.py), so they are not clean
held-out scores. They are what the paper's evaluation measured.

    python -m tools.probe_published
"""

from __future__ import annotations

import numpy as np

from wisense.csi import KEPT_SUBCARRIERS, PROBLEM_KEPT
from wisense.data import RAW_DIR, load
from wisense.evaluate import evaluate_models

PUBLISHED = RAW_DIR.parents[1] / "Embedded_Model"
MODELS = (("small", "small_model_complete_EMD_LSTM.keras", "small_model_complete_EMD_LSTM_uint8.tflite"),
          ("large", "model_complete_synth_IMF.keras", "model_complete_EMD_LSTM_uint8.tflite"))


def main() -> None:
    from tensorflow import keras

    _, tst = load().split()
    for arch, kfile, tfile in MODELS:
        model = keras.models.load_model(PUBLISHED / kfile, compile=False)
        print(f"\n=== published {arch}: {kfile}  ({model.count_params():,} params)")

        p = model.predict(tst.x.transpose(0, 2, 1).reshape(-1, tst.x.shape[1], 1),
                          batch_size=256, verbose=0).reshape(len(tst.y), 52, -1)
        acc = (p.argmax(-1) == tst.y[:, None]).mean(0)
        print("accuracy per kept subcarrier (raw index: acc), * = excluded as problematic:")
        print("  " + "  ".join(f"{KEPT_SUBCARRIERS[k]}{'*' if k in PROBLEM_KEPT else ''}:{a:.2f}"
                               for k, a in enumerate(acc)))
        worst = np.argsort(acc)[:len(PROBLEM_KEPT)]
        print(f"  five worst: raw {sorted(KEPT_SUBCARRIERS[k] for k in worst)}  "
              f"(PROBLEM_KEPT is raw {sorted(KEPT_SUBCARRIERS[k] for k in PROBLEM_KEPT)})")

        tfl = PUBLISHED / tfile
        model.compile()
        res = evaluate_models(model, tfl.read_bytes() if tfl.exists() else None)
        for kind, scores in res.items():
            if isinstance(scores, dict):
                print(f"  {kind:6s} " + "   ".join(f"{m} {r['accuracy']:.4f}" for m, r in scores.items()))
        if "uint8_bytes" in res:
            print(f"  released uint8 file: {res['uint8_bytes']:,} B")


if __name__ == "__main__":
    main()
