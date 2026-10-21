"""Copy the PlatformIO build files into the authors' collecting-tool directory.

The tool's sources stay in the authors' clone; platformio.ini and
sdkconfig.defaults are added beside them, and reset_button.c is patched so the
configuration is only erased by a deliberate 3 s press of BOOT -- the
original erased it on any glitch of GPIO0, which USB serial activity causes.
Existing files are never overwritten silently -- pass --force to replace them.

    python -m tools.install_collecting_tool
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from wisense.data import RAW_DIR, ROOT

SRC = ROOT / "firmware" / "collecting_tool"
DEST = RAW_DIR.parents[1] / "ESP32_CSI_Web_Collecting_Tool"
FILES = ("platformio.ini", "sdkconfig.defaults")
# Patched sources: the authors' original is kept beside it as <name>.orig.
PATCHES = ("main/components/reset_button.c",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (DEST / "main" / "main.c").exists():
        raise SystemExit(f"collecting tool not found at {DEST}")
    for name in FILES:
        target = DEST / name
        if target.exists() and target.read_bytes() != (SRC / name).read_bytes() and not args.force:
            raise SystemExit(f"{target} exists and differs; rerun with --force to replace it")
        shutil.copyfile(SRC / name, target)
        print(f"installed {target}")
    for name in PATCHES:
        target, orig = DEST / name, DEST / (name + ".orig")
        if not orig.exists():
            shutil.copyfile(target, orig)
        shutil.copyfile(SRC / name, target)
        print(f"patched {target} (original kept as {orig.name})")
    print(f"\nnext:  cd \"{DEST}\"  &&  pio run -t upload -t monitor")


if __name__ == "__main__":
    main()
