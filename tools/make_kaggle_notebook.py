"""Write gpu/kaggle_train.ipynb -- the Kaggle GPU training notebook.

    python -m tools.make_kaggle_notebook

Kept as a generator so the notebook is reviewable as plain Python and every
code cell is syntax-checked before it is written (a typo in a cell otherwise
only shows up as a failed run on Kaggle).
"""

from __future__ import annotations

import ast
import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "gpu" / "kaggle_train.ipynb"

INTRO = """# WISENSE -- training on a Kaggle GPU

Trains the LSTM-Embedded DenseNet with the paper's settings (Adam 1e-4, batch 32,
300 full epochs, per-subcarrier samples). Edit **RUNS** below: each run is
(arch, protocol, seed). One run per GPU trains in parallel ("GPU T4 x2" -> two).

* `protocol="clean"` -- synthetic samples from training subjects only (subject-independent score).
* `protocol="paper"` -- synthetic samples from all 22 subjects, as the authors' scripts do (leaky).

The input dataset holds `amplitude.npz`, the IMF file for the protocol (`imfs_train.npz`
or `imfs_paper.npz`) and the `wisense/` code. Each run stops cleanly before the 11-hour
guard; attach this notebook's own output as an input and run again to resume."""

CONFIG = '''RUNS = [("small", "clean", 0), ("small", "clean", 1)]   # (arch, protocol, seed)
MAX_HOURS = 11
'''

SETUP = '''import glob, json, os, shutil, subprocess, sys, time

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None

IMF = {"clean": "imfs_train.npz", "paper": "imfs_paper.npz"}
amp, code_py = find("amplitude.npz"), find("wisense/models.py")
assert amp and code_py, "attach the dataset with amplitude.npz and wisense/"
DATA = os.path.dirname(amp)
for _, protocol, _ in RUNS:
    assert os.path.exists(f"{DATA}/{IMF[protocol]}"), f"{IMF[protocol]} must sit next to amplitude.npz"

WORK = "/kaggle/working/WISENSE"
shutil.copytree(os.path.dirname(code_py), f"{WORK}/wisense", dirs_exist_ok=True)
os.environ.update(WISENSE_CACHE=DATA, TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1")

def run_name(arch, protocol, seed):
    return f"{arch}{'_paper' if protocol == 'paper' else ''}_s{seed}"

for arch, protocol, seed in RUNS:
    name = run_name(arch, protocol, seed)
    out = f"{WORK}/artifacts/{name}"
    prev = find(f"{name}/model.keras")        # this notebook's previous output, if attached
    if prev and not os.path.exists(f"{out}/model.keras"):
        shutil.copytree(os.path.dirname(prev), out, dirs_exist_ok=True)
        print("resuming", name, "from", prev)
'''

CHECK = '''import tensorflow as tf, keras
gpus = tf.config.list_physical_devices("GPU")
print("tensorflow", tf.__version__, "| keras", keras.__version__, "| GPUs:", len(gpus))
assert int(keras.__version__.split(".")[0]) >= 3, "needs Keras 3"
assert gpus, "no GPU -- set Accelerator to GPU T4 x2"
'''

TRAIN = '''def train(arch, protocol, seed, gpu):
    name = run_name(arch, protocol, seed)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = open(f"{WORK}/{name}.log", "a")
    cmd = [sys.executable, "-u", "-m", "wisense.train", "--arch", arch, "--protocol", protocol,
           "--seed", str(seed), "--resume", "--max-hours", str(MAX_HOURS),
           "--out", f"{WORK}/artifacts/{name}"]
    return name, subprocess.Popen(cmd, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)

n_gpu = len(gpus)
pending, failed = list(RUNS), []
while pending:
    batch, pending = pending[:n_gpu], pending[n_gpu:]
    procs = [train(*r, gpu=i) for i, r in enumerate(batch)]
    while any(p.poll() is None for _, p in procs):
        time.sleep(300)
        for name, _ in procs:                     # progress: last epoch line of each run
            lines = [l for l in open(f"{WORK}/{name}.log") if l.startswith("Epoch ")]
            print(time.strftime("%H:%M"), name, lines[-1].strip() if lines else "starting", flush=True)
    failed += [name for name, p in procs if p.returncode != 0]
    for name, p in procs:
        if p.returncode != 0:
            print(name, "FAILED -- last log lines:")
            print("".join(open(f"{WORK}/{name}.log").readlines()[-30:]))
print("failed runs:", failed or "none")
'''

FINISH = '''NL = chr(10)

def py(*args):
    r = subprocess.run([sys.executable, "-u", "-m", *args], cwd=WORK, capture_output=True, text=True)
    print(NL.join(l for l in r.stdout.splitlines() if not l.startswith(("WARNING", "I0000", "W0000"))))
    if r.returncode:
        print(r.stderr[-3000:])

for arch, protocol, seed in RUNS:
    name = run_name(arch, protocol, seed)
    out = f"{WORK}/artifacts/{name}"
    meta = json.load(open(f"{out}/train.json"))
    print(NL + f"===== {name}: {meta['epochs_done']} / {meta['epochs']} epochs")
    if meta["epochs_done"] >= meta["epochs"]:
        py("wisense.export", out, "--no-firmware")
        py("wisense.evaluate", out)
    else:
        print("not finished -- attach this notebook's output as an input and run again to resume")
'''


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src}
    if kind == "code":
        ast.parse(src)                     # refuse to write a notebook that cannot run
        c.update(execution_count=None, outputs=[])
    return c


def main() -> None:
    nb = {"cells": [cell("markdown", INTRO)] + [cell("code", s) for s in (CONFIG, SETUP, CHECK, TRAIN, FINISH)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
