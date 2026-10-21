"""Serial wire-format parsing for host/csi_capture.py."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from csi_capture import AsciiParser, BinaryParser, Fields  # noqa: E402


def frame(payload: np.ndarray, fields: Fields = Fields()) -> bytes:
    out = b"CSI_DATA,"
    if fields.mac:
        out += bytes(6)
    if fields.rssi:
        out += struct.pack("<b", -40)
    if fields.timestamp:
        out += struct.pack("<I", 123456)
    if fields.len:
        out += struct.pack("<H", len(payload))
    return out + payload.astype(np.int8).tobytes()


def test_binary_frames_split_across_reads_and_log_noise():
    a = np.arange(-64, 64)
    b = np.full(128, -3)
    stream = b"I (55) boot: noise CSI_DA\n" + frame(a) + b"junk" + frame(b)
    p = BinaryParser()
    got = []
    for i in range(0, len(stream), 7):     # arbitrary chunking
        got += p.feed(stream[i:i + 7])
    assert len(got) == 2
    assert np.array_equal(got[0], a) and np.array_equal(got[1], b)


def test_binary_optional_fields_are_skipped():
    f = Fields.parse("mac,rssi,timestamp,len")
    p = BinaryParser(f)
    got = p.feed(frame(np.arange(128) - 64, f))
    assert len(got) == 1 and got[0][0] == -64


def test_binary_non_lltf_length_is_dropped():
    p = BinaryParser()
    got = p.feed(frame(np.zeros(384)) + frame(np.ones(128)))
    assert len(got) == 1 and p.dropped == 1 and got[0][0] == 1


def test_ascii_line():
    vals = list(range(-64, 64))
    line = b"CSI_DATA,-40,1,[" + ",".join(map(str, vals)).encode() + b"]\n"
    got = AsciiParser().feed(b"garbage\n" + line)
    assert len(got) == 1 and list(got[0]) == vals


def test_fields_rejects_typo():
    with pytest.raises(ValueError):
        Fields.parse("len,rsi")
