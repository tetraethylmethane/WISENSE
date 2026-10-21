"""Live dashboard: see the Wi-Fi signal, the motion it carries, and the model's call.

    python host/live_view.py --port COM7 --model artifacts/small_paper/model_uint8.tflite
    python host/live_view.py --simulate --model ...          # replay the dataset, no boards
    python host/live_view.py --simulate --snapshot out.png --seconds 25   # headless check

Three panels, updated several times a second:

  1. CSI amplitude, last ~17 s x 47 subcarriers, scrolling. Movement between the
     boards shows as vertical streaks; an empty, still room is a flat texture.
  2. Motion energy -- mean packet-to-packet change, normalised per subcarrier --
     over the last 60 s. Press B with the room empty to set the baseline line.
     This is the setup check: walking between the boards should sit clearly
     above the baseline. If it does not, the boards' placement is the problem,
     not the model.
  3. The model's per-class confidence for the current 850-packet window (needs
     ~17 s of data before the first call), refreshed once per second.

The window is the same one the model classifies; the subcarrier selection and
quantisation are the device's (features.select), so what you see is what the
ESP32-S3 firmware would act on. Serial frames are read on a background thread,
so drawing never drops packets.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wisense.csi import GOOD_KEPT, TIMESTEPS, amplitude, KEPT_SUBCARRIERS  # noqa: E402
from wisense.data import ACTIVITIES, ACTIVITY_NAMES  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"             # validated categorical slots 1-2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD_RAW = [KEPT_SUBCARRIERS[k] for k in GOOD_KEPT]   # raw indices of the 47
ENERGY_WINDOW = 50                                   # packets per motion sample (~1 s)
HISTORY_S = 60


def motion_energy(block: np.ndarray) -> float:
    """Mean |a[t+1] - a[t]| / mean(a), averaged over subcarriers. (T, S) -> scalar."""
    if len(block) < 2:
        return 0.0
    return float((np.abs(np.diff(block, axis=0)).mean(0) / (block.mean(0) + 1e-6)).mean())


class Classifier:
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp, self.out = self.interp.get_input_details()[0], self.interp.get_output_details()[0]
        self.scale, self.zero = self.inp["quantization"]

    def __call__(self, window52: np.ndarray) -> np.ndarray:
        from wisense.features import select
        x = select(window52[None])[0]
        q = np.clip(np.round(x / self.scale + self.zero), 0, 255).astype(np.uint8)
        self.interp.set_tensor(self.inp["index"], q[None, :, None])
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0].astype(float) / 255.0


def reader(source, frames: queue.Queue, stop: threading.Event) -> None:
    for _, csi in source:
        if stop.is_set():
            return
        frames.put(csi)


def main() -> None:
    from csi_capture import Fields, find_rx_port, serial_frames, simulated_frames

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="auto", help="COMx, or auto (default): find the Rx")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fields", default="len")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--model", default=None, help="uint8 .tflite; omit to show signal only")
    ap.add_argument("--snapshot", default=None, help="render to this file after --seconds and exit")
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--smooth", type=int, default=5,
                    help="average the last N once-per-second calls (steadier display)")
    ap.add_argument("--min-conf", type=float, default=0.6,
                    help="below this averaged confidence the call shows as Unsure")
    args = ap.parse_args()

    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.simulate and args.port.lower() == "auto":
        args.port = find_rx_port(args.baud)   # before any timed step starts
    if args.simulate:
        source = simulated_frames()
    else:
        source = serial_frames(args.port, args.baud, False, Fields.parse(args.fields))
    clf = Classifier(args.model) if args.model else None

    frames: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=reader, args=(source, frames, stop), daemon=True).start()

    window: deque[np.ndarray] = deque(maxlen=TIMESTEPS)       # (52,) amplitude rows
    energy_t: deque[float] = deque(maxlen=HISTORY_S)
    energy_v: deque[float] = deque(maxlen=HISTORY_S)
    recent_probs: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    state = {"since_e": 0, "since_c": 0, "probs": None, "baseline": None,
             "t0": time.time(), "n": 0, "rate": 0.0}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1.3, 1], hspace=0.38, wspace=0.25)
    ax_hm, ax_en, ax_cl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    title = fig.suptitle("WISENSE live -- waiting for CSI", color=INK, fontsize=12, x=0.02, ha="left")

    hm = ax_hm.imshow(np.zeros((len(GOOD_KEPT), TIMESTEPS)), aspect="auto", cmap="Blues",
                      origin="lower", vmin=0, vmax=40, extent=(-TIMESTEPS / 50, 0, 0, len(GOOD_KEPT)))
    ax_hm.set_title("CSI amplitude (movement shows as vertical streaks)", loc="left", color=INK)
    ax_hm.set_xlabel("seconds ago")
    ax_hm.set_ylabel("subcarrier")
    ticks = [0, 12, 24, 36, 46]
    ax_hm.set_yticks([t + 0.5 for t in ticks], [str(GOOD_RAW[t]) for t in ticks])
    fig.colorbar(hm, ax=ax_hm, fraction=0.025, pad=0.01, label="amplitude")

    (en_line,) = ax_en.plot([], [], color=BLUE, lw=1.8, label="motion energy")
    base_line = ax_en.axhline(np.nan, color=INK2, lw=1, ls="--", label="empty-room baseline (B)")
    ax_en.set_title("Motion energy  --  press B with the room empty to set the baseline",
                    loc="left", color=INK)
    ax_en.set_xlim(-HISTORY_S, 0)
    ax_en.set_xlabel("seconds ago")
    ax_en.grid(True, color=GRID, lw=0.6)
    ax_en.legend(loc="upper left", frameon=False, fontsize=8)
    en_text = ax_en.text(0.99, 0.95, "", transform=ax_en.transAxes, ha="right", va="top", color=INK)

    labels = [ACTIVITY_NAMES[a] for a in ACTIVITIES]
    bars = ax_cl.barh(labels, [0] * len(labels), color=BLUE, height=0.6)
    ax_cl.set_xlim(0, 1)
    ax_cl.invert_yaxis()
    ax_cl.set_title(f"Model confidence (avg. of last {args.smooth} calls)", loc="left", color=INK)
    ax_cl.grid(True, axis="x", color=GRID, lw=0.6)
    call_text = ax_cl.text(0.5, -0.07, "", transform=ax_cl.transAxes, ha="center", va="top",
                           fontsize=15, color=INK, weight="bold")
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.07, right=0.97)
    if clf is None:
        call_text.set_text("no model loaded")

    def on_key(ev):
        if ev.key in ("b", "B") and energy_v:
            state["baseline"] = float(np.median(list(energy_v)[-5:]))

    fig.canvas.mpl_connect("key_press_event", on_key)

    def drain():
        while True:
            try:
                csi = frames.get_nowait()
            except queue.Empty:
                break
            amp = amplitude(np.asarray(csi)[None])[0, list(KEPT_SUBCARRIERS)]
            window.append(amp)
            state["n"] += 1
            state["since_e"] += 1
            state["since_c"] += 1
            if state["since_e"] >= ENERGY_WINDOW and len(window) >= ENERGY_WINDOW:
                state["since_e"] = 0
                recent = np.array(list(window)[-ENERGY_WINDOW:])[:, list(GOOD_KEPT)]
                energy_t.append(time.time())
                energy_v.append(motion_energy(recent))
            if clf and len(window) == TIMESTEPS and state["since_c"] >= ENERGY_WINDOW:
                state["since_c"] = 0
                recent_probs.append(clf(np.array(window)))
                state["probs"] = np.mean(recent_probs, axis=0)

    def update(_=None):
        drain()
        elapsed = time.time() - state["t0"]
        state["rate"] = state["n"] / elapsed if elapsed > 0 else 0
        if window:
            w = np.array(window)[:, list(GOOD_KEPT)].T
            img = np.zeros((len(GOOD_KEPT), TIMESTEPS))
            img[:, -w.shape[1]:] = w
            hm.set_data(img)
        if energy_v:
            now = time.time()
            en_line.set_data([t - now for t in energy_t], list(energy_v))
            top = max(max(energy_v), state["baseline"] or 0) * 1.25 + 1e-3
            ax_en.set_ylim(0, top)
            ratio = ""
            if state["baseline"]:
                base_line.set_ydata([state["baseline"]] * 2)
                ratio = f"   now / baseline = {energy_v[-1] / state['baseline']:.2f}x"
            en_text.set_text(f"{energy_v[-1]:.3f}{ratio}")
        if state["probs"] is not None:
            p = state["probs"]
            k = int(np.argmax(p))
            for i, b in enumerate(bars):
                b.set_width(p[i])
                b.set_color(ORANGE if i == k else BLUE)
            if p[k] < args.min_conf:
                call_text.set_text(f"Unsure  (best: {labels[k]} {p[k]:.0%})")
                call_text.set_color(INK2)
            else:
                call_text.set_text(f"{labels[k]}  ({p[k]:.0%})")
                call_text.set_color(INK)
        fill = len(window) / TIMESTEPS
        status = f"{state['rate']:.0f} packets/s" + ("" if fill >= 1 else f"  --  filling window {fill:.0%}")
        title.set_text(f"WISENSE live  --  {status}")
        return []

    if args.snapshot:
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(0.2)
            update()
        state["baseline"] = state["baseline"] or (float(np.median(energy_v)) if energy_v else None)
        update()
        fig.savefig(args.snapshot, dpi=110)
        print(f"saved {args.snapshot}: {state['n']} frames, {state['rate']:.1f} pps")
        stop.set()
        return

    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, update, interval=250, cache_frame_data=False)  # noqa: F841
    plt.show()
    stop.set()


if __name__ == "__main__":
    main()
