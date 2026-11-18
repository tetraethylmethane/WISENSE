# Live demo: recognising real activities in the room

A step-by-step guide for filming a person falling, sitting down and so on, with the
model classifying the Wi-Fi signal live. It assumes both ESP32s already run the
collecting tool and are configured as Tx and Rx (README, *Hardware*, steps 1–3).

## Read this first

**Run as-is, the model will mostly be wrong in your room.** On people it has never seen it
scores 52.6%, and in a new room 33.7% (chance is 20%). There are two honest ways to demo:

| | **A. Calibrate first (recommended)** | **B. Run as-is** |
|---|---|---|
| What you do | Record the demo person in the demo room, fine-tune on it | Run the released model live |
| Expect | Mostly correct | Mostly wrong |
| Say on screen | "Calibrated to this person and room" | "This is our 52.6% / 33.7% finding, live" |

Also:

- **Each prediction takes ~17 s.** The model needs 850 packets (17 s at 50 packets/s)
  before it answers. Treat every activity as a 20 s trial: start, do the movement, wait.
- **Inference runs on the laptop**, not the ESP32-S3.
- **Use a mattress or gym mat** for falls and lying down.
- **Consent:** recording anyone other than yourselves needs their informed consent, and
  possibly ethics approval.
- Nobody else should move near the boards during a trial.

## Steps

Commands are for PowerShell, run from the repository root.

### 0. One-time setup

```powershell
python -m pip install -r requirements.txt
```

> If the repo or your virtual environment is inside OneDrive and imports hang for
> minutes, the files are "online-only". Put the virtual environment outside OneDrive
> (for example `C:\venvs\wisense`) and use that `python`.

### 1. Plug the Rx board in over USB and check the stream

```powershell
python host/csi_capture.py stats            # expect ~50 packets/s; add --port COM5 if auto fails
```

### 2. Place the boards

Put them 2–3 m apart, at waist height, with the activity area between them. Then:

```powershell
python host/placement_check.py
```

Follow the prompts (stay still, then walk across the line between the boards). Move the
boards and repeat until it says **GOOD**. Skipping this is why our first room recordings
failed: the movement level read ~0.02 even while walking, so the link was not seeing the
person.

### 3. Record calibration data (option A only)

One command per activity: 15 reps × 20 s each (~5 min). Do each movement the way you will
do it in the demo, in the same spot.

```powershell
python host/csi_capture.py record --activity FA --subject s23   # fall
python host/csi_capture.py record --activity GU --subject s23   # get up
python host/csi_capture.py record --activity LD --subject s23   # lie down
python host/csi_capture.py record --activity SD --subject s23   # sit down
python host/csi_capture.py record --activity WA --subject s23   # walk
```

Add `--reps 10` to make it quicker. Recordings land in `data/recordings/` in the dataset's
format.

### 4. Fine-tune on those recordings (option A only)

> **Not written yet.** Planned: a CPU fine-tune of the released model on
> `data/recordings/`, exporting a new 8-bit model, along the lines of
>
> ```powershell
> python -m wisense.finetune data/recordings --out artifacts/calibrated
> ```

### 5. Run the live demo and film it

1. Start a screen recording (OBS, or Windows + Alt + R).
2. Start the phone recording, on a tripod, with the whole activity area in frame.
3. Run the dashboard:

   ```powershell
   # Option A (calibrated)
   python host/live_view.py --model artifacts/calibrated/model_uint8.tflite
   # Option B (as-is)
   python host/live_view.py --model artifacts/small_paper/model_uint8.tflite
   ```

4. **Clap once** where the camera sees it, so the two recordings can be synced.
5. Do each activity once, as a 20 s trial, and wait for the prediction before the next.

> **Planned:** a trial mode for `live_view.py`. A key press starts a 20 s countdown, then
> one large prediction appears. It reads better on camera than the rolling display.

### 6. Edit

Put the phone video and the screen recording side by side, synced on the clap. Add a caption
naming each activity, and the "calibrated" note if you used option A.

## Checklist for the day

- [ ] Both boards powered, Rx on USB, `stats` shows ~50 packets/s
- [ ] `placement_check` says GOOD
- [ ] Mat in place for falls and lying down
- [ ] Consent from everyone being filmed
- [ ] Calibration recorded and fine-tuned (option A)
- [ ] Screen and phone both recording, clap done
- [ ] Room clear of other people moving
