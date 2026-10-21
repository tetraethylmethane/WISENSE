# Third-party notices

## Embedded_WiFi_Sensing (Armenta-Garcia et al.)

<https://github.com/AlbanyArmenta0711/Embedded_WiFi_Sensing>

WISENSE reproduces this work. Its dataset, collection firmware and trained models
are **not** redistributed here; clone the repository next to this one (see README).

Derived from it in this repository:

- `firmware/collecting_tool/main/components/reset_button.c` -- the original file,
  patched so the stored configuration is erased only by a 3 s press of BOOT.
- `firmware/collecting_tool/platformio.ini` and `sdkconfig.defaults` -- build
  configuration for the original firmware (new files, not copies).
- `wisense/csi.py`, `wisense/augment.py`, `wisense/data.py` -- vectorised
  re-implementations of the preprocessing, EEMD recombination and split logic of
  the original scripts.
- `firmware/tests/csi_reference.h` -- 900 packets of one recording from the
  original dataset, frozen as a test fixture.

```
MIT License

Copyright (c) 2025 AlbanyArmenta0711

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
