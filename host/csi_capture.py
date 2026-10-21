"""Capture CSI from the authors' ESP32 CSI Web Collecting Tool over USB serial.

Pairs with the Rx board running ESP32_CSI_Web_Collecting_Tool in Console
informer mode. Writes recordings in exactly the dataset's format -- one line
per packet: host milliseconds, then 128 int8 values (imag, real per
subcarrier) -- so anything recorded here drops straight into wisense.data.

    python host/csi_capture.py stats  --port COM5               # check packet rate
    python host/csi_capture.py record --port COM5 --activity FA --subject s25
    python host/csi_capture.py live   --port COM5 --model artifacts/small/model_uint8.tflite
    python host/csi_capture.py live   --simulate --model ...    # replay dataset, no board

WIRE FORMAT (informer.c)
------------------------
Binary: b"CSI_DATA," then, for each field ticked in the web form and in this
order: mac[6], rssi i8, bandwidth u8, noise floor i8, timestamp u32, antenna
u8, len u16, then len bytes of CSI. Tick at least "CSI length" and "CSI data"
in binary mode -- without the length there is no framing beyond the marker.
ASCII: "CSI_DATA,<fields>,[v0,v1,...]\\n". ESP-IDF log lines share the port;
both parsers resynchronise on the marker and ignore everything else.

Opening the port never toggles DTR/RTS: on DevKit boards that pulses GPIO0,
which the tool treats as its erase-configuration button.

Baud: 115200 is the tool's console default, and is why the paper measured an
85 pps ceiling (115200/10 bytes/s / ~136 bytes per packet). 50 pps fits.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MARKER = b"CSI_DATA,"
CSI_LEN = 128

# (flag name, struct format) in informer.c's emission order.
BINARY_FIELDS = (("mac", "6s"), ("rssi", "<b"), ("bw", "<B"), ("noise", "<b"),
                 ("timestamp", "<I"), ("ant", "<B"), ("len", "<H"))


@dataclass
class Fields:
    """Which optional fields the web form enabled. Must match the board."""
    mac: bool = False
    rssi: bool = False
    bw: bool = False
    noise: bool = False
    timestamp: bool = False
    ant: bool = False
    len: bool = True

    @classmethod
    def parse(cls, spec: str) -> "Fields":
        names = {n.strip() for n in spec.split(",") if n.strip()}
        unknown = names - {n for n, _ in BINARY_FIELDS}
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}")
        return cls(**{n: n in names for n, _ in BINARY_FIELDS})


@dataclass
class BinaryParser:
    """Incremental parser: feed() bytes, get complete 128-byte CSI frames."""
    fields: Fields = field(default_factory=Fields)
    buf: bytearray = field(default_factory=bytearray)
    dropped: int = 0

    def feed(self, data: bytes) -> list[np.ndarray]:
        self.buf += data
        frames = []
        while True:
            start = self.buf.find(MARKER)
            if start < 0:
                del self.buf[:max(0, len(self.buf) - len(MARKER) + 1)]
                return frames
            pos = start + len(MARKER)
            info = {}
            for name, fmt in BINARY_FIELDS:
                if getattr(self.fields, name):
                    size = struct.calcsize(fmt)
                    if len(self.buf) < pos + size:
                        del self.buf[:start]
                        return frames
                    info[name] = struct.unpack_from(fmt, self.buf, pos)[0]
                    pos += size
            n = info.get("len", CSI_LEN)
            if n not in (128, 256, 384) or n > 1024:
                self.dropped += 1            # false marker inside payload; skip it
                del self.buf[:start + 1]
                continue
            if len(self.buf) < pos + n:
                del self.buf[:start]
                return frames
            raw = np.frombuffer(bytes(self.buf[pos:pos + n]), dtype=np.int8)
            del self.buf[:pos + n]
            if n != CSI_LEN:
                self.dropped += 1            # HT-LTF present: not the dataset's LLTF-only layout
                continue
            frames.append(raw.astype(np.int16))


@dataclass
class AsciiParser:
    buf: bytearray = field(default_factory=bytearray)
    dropped: int = 0

    def feed(self, data: bytes) -> list[np.ndarray]:
        self.buf += data
        frames = []
        while b"\n" in self.buf:
            line, _, rest = self.buf.partition(b"\n")
            self.buf = bytearray(rest)
            i = line.find(MARKER)
            if i < 0 or b"[" not in line:
                continue
            try:
                body = line[line.index(b"[") + 1:line.rindex(b"]")].decode("ascii")
                vals = np.array([int(v) for v in body.split(",")], dtype=np.int16)
            except ValueError:
                self.dropped += 1
                continue
            if len(vals) == CSI_LEN:
                frames.append(vals)
            else:
                self.dropped += 1
        return frames


def _open(port: str, baud: int, timeout: float):
    import serial

    # DTR/RTS must stay deasserted *before* the port opens. On DevKit boards
    # they drive EN and GPIO0 through the auto-reset circuit, and GPIO0 is
    # the collecting tool's "erase configuration" button: pyserial's default
    # open pulses it and silently wipes the board's settings.
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, timeout
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def find_rx_port(baud: int = 115200, listen_s: float = 2.0) -> str:
    """The serial port that is actually streaming CSI frames.

    Windows renumbers COM ports when a board is replugged, and the Tx sits on
    a port too, so "the CH340 one" is not a reliable rule. Listen briefly on
    every USB serial port and pick the one carrying CSI_DATA markers.
    """
    from serial.tools import list_ports

    best, best_n = None, 0
    for p in list_ports.comports():
        if "bluetooth" in (p.description or "").lower() or not p.vid:
            continue
        try:
            with _open(p.device, baud, 0.1) as ser:
                ser.reset_input_buffer()
                buf, end = b"", time.time() + listen_s
                while time.time() < end:
                    buf += ser.read(4096)
        except Exception as e:  # port busy or vanished: report, keep looking
            print(f"  {p.device}: cannot open ({e.__class__.__name__})", file=sys.stderr)
            continue
        n = buf.count(MARKER)
        print(f"  {p.device} ({p.description}): {n / listen_s:.0f} CSI frames/s", file=sys.stderr)
        if n > best_n:
            best, best_n = p.device, n
    if best is None:
        sys.exit("No port is streaming CSI. Is the Rx plugged in, and the Tx powered and "
                 "connected to it? (Configured Rx shows Wi-Fi 'CSI_Collection'.)")
    print(f"using {best}", file=sys.stderr)
    return best


def serial_frames(port: str | None, baud: int, ascii_mode: bool, fields: Fields) -> Iterator[tuple[int, np.ndarray]]:
    if not port or port.lower() == "auto":
        port = find_rx_port(baud)
    parser = AsciiParser() if ascii_mode else BinaryParser(fields)
    try:
        ser = _open(port, baud, 0.05)
    except Exception as e:
        sys.exit(f"Cannot open {port}: {e}\nRun without --port (or --port auto) to find the Rx.")
    with ser:
        ser.reset_input_buffer()
        while True:
            chunk = ser.read(ser.in_waiting or 1)
            for f in parser.feed(chunk):
                yield int(time.time() * 1000), f


def simulated_frames(fs: float = 50.0) -> Iterator[tuple[int, np.ndarray]]:
    """Replay held-out-subject recordings at the real packet rate."""
    from wisense.csi import read_raw
    from wisense.data import RAW_DIR, TEST_SUBJECTS, index

    recs = [r for r in index() if r.subject in TEST_SUBJECTS]
    rng = np.random.default_rng()
    while True:
        r = recs[rng.integers(len(recs))]
        print(f"[simulate] replaying {r.path.relative_to(RAW_DIR)}  (truth: {r.activity})", flush=True)
        _, csi = read_raw(r.path)
        for row in csi:
            time.sleep(1 / fs)
            yield int(time.time() * 1000), row


def frame_source(args) -> Iterator[tuple[int, np.ndarray]]:
    if args.simulate:
        return simulated_frames()
    return serial_frames(args.port, args.baud, args.ascii, Fields.parse(args.fields))


def write_recording(path: Path, rows: list[tuple[int, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for ts, csi in rows:
            f.write(f"{ts}," + ",".join(str(int(v)) for v in csi) + "\n")


def cmd_stats(args) -> None:
    t0, n, last = time.time(), 0, time.time()
    for _, _ in frame_source(args):
        n += 1
        if time.time() - last >= 1.0:
            last = time.time()
            print(f"{n / (last - t0):6.1f} pps average, {n} frames", flush=True)


def cmd_record(args) -> None:
    from wisense.data import ACTIVITIES

    # EM (empty room, nobody present) is not in the paper's dataset; it is
    # recorded for cross-environment work, where the model must also learn
    # what "no activity" looks like.
    labels = ACTIVITIES + ("EM",)
    if args.activity not in labels:
        sys.exit(f"--activity must be one of {labels}")
    src = frame_source(args)
    out = Path(args.out) / args.activity / args.subject
    for rep in range(1, args.reps + 1):
        input(f"[{args.activity} {args.subject}] repetition {rep}/{args.reps}: "
              f"press Enter, then perform the activity for {args.seconds} s ")
        rows, t_end = [], time.time() + args.seconds
        for ts, csi in src:
            rows.append((ts, csi))
            if time.time() >= t_end:
                break
        path = out / f"csi_data_{int(time.time())}.txt"
        write_recording(path, rows)
        ok = "ok" if len(rows) >= 850 else "TOO SHORT -- needs >= 850 packets, redo"
        print(f"  {len(rows)} packets ({len(rows) / args.seconds:.1f} pps) -> {path}  [{ok}]")


def cmd_live(args) -> None:
    """Sliding 850-packet window, classified every --hop packets on the host."""
    import tensorflow as tf

    from wisense.csi import TIMESTEPS, preprocess
    from wisense.data import ACTIVITY_NAMES, ACTIVITIES
    from wisense.features import select

    interp = tf.lite.Interpreter(model_path=args.model)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zero = i["quantization"]
    window: deque[np.ndarray] = deque(maxlen=TIMESTEPS)
    since = 0
    for _, csi in frame_source(args):
        window.append(csi)
        since += 1
        if len(window) < TIMESTEPS or since < args.hop:
            continue
        since = 0
        x = select(preprocess(np.stack(window))[None])[0]
        q = np.clip(np.round(x / scale + zero), 0, 255).astype(np.uint8)
        t0 = time.perf_counter()
        interp.set_tensor(i["index"], q[None, :, None])
        interp.invoke()
        p = interp.get_tensor(o["index"])[0].astype(float)
        k = int(p.argmax())
        print(f"{ACTIVITY_NAMES[ACTIVITIES[k]]:9s} {p[k] / 255:.2f}  "
              f"({(time.perf_counter() - t0) * 1e3:.1f} ms host)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("stats", cmd_stats), ("record", cmd_record), ("live", cmd_live)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--port", default="auto", help="COMx, or auto (default): find the Rx")
        p.add_argument("--baud", type=int, default=115200)
        p.add_argument("--ascii", action="store_true", help="board set to ASCII data mode")
        p.add_argument("--fields", default="len",
                       help="optional fields ticked in the web form, comma-separated: "
                            "mac,rssi,bw,noise,timestamp,ant,len (binary mode)")
        p.add_argument("--simulate", action="store_true", help="replay the dataset instead of a board")
        if name == "record":
            p.add_argument("--activity", required=True)
            p.add_argument("--subject", required=True)
            p.add_argument("--reps", type=int, default=15)
            p.add_argument("--seconds", type=float, default=20.0)
            p.add_argument("--out", default="data/recordings")
        if name == "live":
            p.add_argument("--model", required=True)
            p.add_argument("--hop", type=int, default=50, help="packets between classifications")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
