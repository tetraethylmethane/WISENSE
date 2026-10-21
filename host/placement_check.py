"""Guided board-placement check: is the Tx-Rx link actually sensing you?

    python host/placement_check.py --port COM7

Two timed steps, then a plain verdict:
  1. keep the area empty and still   -> baseline movement level
  2. walk back and forth ACROSS the line between the boards -> walking level
Move the boards and repeat until it says GOOD, then record.

MOVEMENT LEVEL
--------------
Each packet's amplitudes are divided by their mean over subcarriers (cancels
the receiver's automatic gain jumps), smoothed over 10 packets (~0.2 s, keeps
body-speed changes, drops packet-rate noise), and the spread over time is
averaged across the 47 subcarriers. Reference values from the paper's own
dataset: every activity there sits at 0.09-0.19; a still room reads ~0.02.
Our first room recordings mostly read ~0.02 even while walking -- the link was
not seeing the subject -- which is what this check exists to catch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wisense.csi import GOOD_KEPT, KEPT_SUBCARRIERS, amplitude  # noqa: E402

LAB_LEVEL = 0.08          # lower end of the paper's dataset, with margin
SMOOTH = 10


def movement_level(rows: np.ndarray) -> float:
    """(T, 64)-derived (T, 47) amplitude -> scalar movement level."""
    if len(rows) < SMOOTH * 3:
        return 0.0
    a = rows / (rows.mean(axis=1, keepdims=True) + 1e-6)
    ker = np.ones(SMOOTH) / SMOOTH
    sm = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="valid"), 0, a)
    return float(sm.std(axis=0).mean())


def bar(v: float, full: float = 0.2, width: int = 30) -> str:
    n = int(min(v / full, 1.0) * width)
    return "#" * n + "." * (width - n)


def measure(frames, seconds: float, label: str) -> float:
    rows, t_end, last = [], time.time() + seconds, 0.0
    for _, csi in frames:
        amp = amplitude(np.asarray(csi)[None])[0][list(KEPT_SUBCARRIERS)]
        rows.append(amp[list(GOOD_KEPT)])
        now = time.time()
        if now - last >= 1.0 and len(rows) > 100:
            last = now
            v = movement_level(np.array(rows[-100:]))
            print(f"\r  {label:8s} [{bar(v)}] {v:.3f}   {t_end - now:4.1f}s left ", end="", flush=True)
        if now >= t_end:
            break
    print()
    return movement_level(np.array(rows))


def verdict(still: float, walk: float) -> str:
    ratio = walk / still if still > 0 else float("inf")
    head = f"\n  empty/still: {still:.3f}    walking: {walk:.3f}    ratio: {ratio:.1f}x\n"
    if walk >= LAB_LEVEL and ratio >= 3:
        return head + "  GOOD -- the link sees you as clearly as in the paper's lab. Record here.\n"
    if ratio >= 2:
        return head + ("  OK -- movement is visible but weaker than the paper's lab.\n"
                       "  Try: boards closer (2 m), same height as your waist, or walk right\n"
                       "  through the straight line between them.\n")
    return head + ("  BAD -- the link barely notices you. Try, one at a time:\n"
                   "   - stand the boards so their antennas point UP, same height (~1 m)\n"
                   "   - 2-3 m apart with nothing (laptop, metal, wall corner) between them\n"
                   "   - walk ACROSS the line between them, not beside or behind a board\n"
                   "   - keep the Rx away from the laptop and other USB/radio gear\n")


def main() -> None:
    from csi_capture import Fields, find_rx_port, serial_frames, simulated_frames

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="auto", help="COMx, or auto (default): find the Rx")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    if not args.simulate and args.port.lower() == "auto":
        args.port = find_rx_port(args.baud)   # before any timed step starts
    if args.simulate:
        frames = simulated_frames()
    else:
        frames = serial_frames(args.port, args.baud, False, Fields.parse("len"))

    print(__doc__.split("MOVEMENT LEVEL")[0].split("\n\n", 1)[1])
    while True:
        input("STEP 1: leave the area between the boards empty and keep still. Press Enter... ")
        time.sleep(2)
        still = measure(frames, args.seconds, "still")
        input("STEP 2: walk back and forth ACROSS the line between the boards. Press Enter... ")
        walk = measure(frames, args.seconds, "walking")
        print(verdict(still, walk))
        if input("Move the boards and press Enter to test again, or type q to quit: ").strip().lower() == "q":
            break


if __name__ == "__main__":
    main()
