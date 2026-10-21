"""Hold a board's serial port open and log everything, without touching DTR/RTS.

For debugging the collecting tool: its log lines and binary CSI frames share
the port. Text lines go to the log as they are; each second also gets a
count of CSI frames seen. Opened once and kept open, so the auto-reset lines
are not pulsed again (see csi_capture.py on why that matters).

    python host/serial_log.py --port COM7 --out data/rx_serial.log --seconds 900
"""

from __future__ import annotations

import argparse
import time

import serial


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=900)
    args = ap.parse_args()

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = args.port, args.baud, 0.1
    s.dtr = False
    s.rts = False
    s.open()
    t0 = last = time.time()
    frames, pending = 0, b""
    with s, open(args.out, "a", encoding="utf-8") as log:
        log.write(f"--- opened {args.port} at {time.strftime('%H:%M:%S')}\n")
        while time.time() - t0 < args.seconds:
            pending += s.read(4096)
            frames += pending.count(b"CSI_DATA,")
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                text = raw.decode("latin-1")
                if text[:3] in ("I (", "W (", "E (") or "Button" in text or "rst:" in text:
                    log.write(f"{time.strftime('%H:%M:%S')} {text.rstrip()}\n")
            if len(pending) > 8192:
                pending = pending[-64:]
            if time.time() - last >= 1.0:
                log.write(f"{time.strftime('%H:%M:%S')} [csi frames this second: {frames}]\n")
                frames, last = 0, time.time()
                log.flush()


if __name__ == "__main__":
    main()
