# hailo-rpi5-custom-yolo

[![CI](https://github.com/baig1102/hailo-rpi5-custom-yolo/actions/workflows/ci.yml/badge.svg)](https://github.com/baig1102/hailo-rpi5-custom-yolo/actions/workflows/ci.yml)

Set up a Hailo-8L on a Raspberry Pi 5 and run **your own** custom-trained YOLO model,
with small tools that catch the common mistakes before they cost you a day.

Written from my own experience deploying a custom model on a Pi 5 + Hailo-8L.
Contains no employer code or data.

> **v0.1.0: unit-tested, hardware validation pending.** Every piece of logic is tested
> in CI without Hailo hardware. The setup, the API path and the lessons come from a
> real deployment, but this repo's code hasn't run on a Pi yet. See
> [what is verified](#what-is-verified).

## The problem

Hailo's examples run Hailo's own models well. Running a model you trained yourself is
where it gets hard, and the docs say little about it:

- `hailortcli` reports `Board Name: Hailo-8` on a Hailo-8L. What decides whether a HEF
  loads is the *device architecture*, not the board name.
- HailoRT, the PCIe driver, the firmware and the Python bindings have to match, and the
  HEF must come from the matching compiler release. A mismatch gives errors that don't
  point at the cause.
- If the model's input size and the image size in the built-in NMS config differ, every
  box comes out scaled wrong, and it's tempting to "fix" that with a hand-measured factor.
- Picamera2's colour format names are the reverse of the array order you get, and the NMS
  output puts y before x.

This repo turns those lessons into guides plus a few checks you can run on the Pi.

## What's in it

| Part | What it does |
|---|---|
| [`doctor`](src/hailo_yolo/doctor.py) | Checks that HailoRT, the driver (installed and loaded), the firmware (installed and running) and the Python bindings match; that a device answers; that `hailo_platform` imports. Says what to do for each problem. |
| [`hefcheck`](src/hailo_yolo/hefcheck.py) | Reads `hailortcli parse-hef` output. Catches a HEF compiled for the wrong chip, a missing built-in NMS, and an NMS image size that differs from the input size. |
| [`preprocess`](src/hailo_yolo/preprocess.py), [`postprocess`](src/hailo_yolo/postprocess.py) | Letterbox like Ultralytics and back; Picamera2 frames to RGB; both HailoRT NMS layouts to pixel boxes on the original image. |
| [`engine`](src/hailo_yolo/engine.py) | Runs a HEF on the legacy `InferVStreams` API: absolute-path check, retry while the chip is busy, fixed shutdown order. |
| [`examples/`](examples) | One image, a live camera loop (one worker owns camera and engine), and a systemd unit. |
| [`docs/`](docs) | Pi setup, running your model, a compile overview, and troubleshooting. |

## Quick start (on the Pi)

1. Install Hailo's software and reboot ([setup guide](docs/01-pi-setup.md)):

   ```bash
   sudo apt install dkms
   sudo apt install hailo-all
   sudo reboot
   ```

2. Check the setup. This uses the system Python and needs no install:

   ```bash
   git clone https://github.com/baig1102/hailo-rpi5-custom-yolo.git
   cd hailo-rpi5-custom-yolo
   ./scripts/doctor.sh
   ```

3. Install the package into a venv that can still see `apt`'s `hailo_platform`:

   ```bash
   sudo apt install python3-opencv python3-picamera2
   python3 -m venv --system-site-packages .venv
   .venv/bin/pip install -e .
   ```

4. Check your HEF, then run it on an image ([run your model](docs/02-run-your-model.md)):

   ```bash
   .venv/bin/hailo-yolo-hefcheck /home/pi/models/my_model.hef --device-arch HAILO8L
   .venv/bin/python examples/infer_image.py /home/pi/models/my_model.hef photo.jpg out.jpg
   ```

From your own code:

```python
from hailo_yolo.engine import HefEngine

with HefEngine("/home/pi/models/my_model.hef") as engine:
    detections = engine.detect(rgb_image, score_threshold=0.5)  # any size, RGB, uint8
for d in detections:
    print(d.class_id, round(d.score, 2), d.x1, d.y1, d.x2, d.y2)  # pixels on rgb_image
```

## Guides

1. [Set up the Pi](docs/01-pi-setup.md): install, verify, run the doctor
2. [Run your model](docs/02-run-your-model.md): check the HEF, run it, read the output
3. [From your model to a HEF](docs/03-compile-overview.md): overview only, not yet a tested recipe
4. [Troubleshooting](docs/troubleshooting.md): 13 problems, each as symptom → cause → fix

## The setup this is based on

The setup I deployed my own model on, and the only "tested on hardware" facts here:

| Layer | Version |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1 |
| OS | Raspberry Pi OS Bookworm (Debian 12), 64-bit, kernel 6.12 |
| Accelerator | Hailo-8L (reports `Board Name: Hailo-8`, `Device Architecture: HAILO8L`) |
| `hailo-all` / HailoRT / `hailortcli` / firmware / PCIe driver / `python3-hailort` | 4.20.0 |
| TAPPAS core | 3.31.0 |
| Python | 3.11.2 (system), `hailo_platform` from `apt`, numpy 1.26.4 |
| Camera | Picamera2 0.3.31 |
| PCIe | `dtparam=pciex1_gen=3` (optional) |
| API that worked | `HEF` → `VDevice` → `ConfigureParams.create_from_hef(..., PCIe)` → `configure` → `activate` → `InferVStreams(..., tf_nms_format=True)` |

Raspberry Pi's AI documentation now targets Raspberry Pi OS **Trixie**, which ships
HailoRT 4.23. The doctor knows Trixie's package names, but nothing here has been tested
against 4.23. Hailo's software isn't included in this repo; get it from Raspberry Pi's and
Hailo's official sources.

## What is verified

| Part | How it is verified | On hardware |
|---|---|---|
| Letterbox and its inverse | Unit tests; geometry checked against Ultralytics' `LetterBox` source | Pending |
| NMS parsing, both layouts | Unit tests; layouts and field order checked against the HailoRT 4.20 source | Pending |
| Picamera2 colour order | Unit tests; checked against the Picamera2 manual, Appendix A | Pending |
| `hefcheck` | Unit tests on synthetic `parse-hef` output rebuilt from the HailoRT 4.20 source | Pending |
| `doctor` | Unit tests with faked commands; package names from Raspberry Pi's repository indexes; run on a machine without Hailo software | Pending |
| `engine` | Unit tests against a fake `hailo_platform` that copies the 4.20 call signatures | Pending (the API path itself worked on my setup) |
| Examples, systemd unit | End-to-end tests with a fake camera and a fake `hailo_platform` | Pending |
| Compile overview | Concept only, **not a tested recipe** | Not applicable |
| Troubleshooting | Each entry marked "my setup" or cited to the HailoRT source by file and line | Experience, not this code |

CI runs ruff, `mypy --strict` and 129 tests on Python 3.11 and 3.12. Every Hailo-specific
claim in the code and docs names its source.

## Limitations

- **No hardware run of this code yet.** Real `parse-hef` and `fw-control identify`
  output may differ in details the source doesn't show.
- **HailoRT 4.20 only.** Trixie's 4.23 isn't tested.
- **Detection only.** The code expects one input and one NMS output: no segmentation or
  pose, and no raw-output HEFs.
- **Legacy `InferVStreams` API.** `InferModel` is newer, but its Python API changed between
  HailoRT versions and isn't verified here, so v0.1 has no `InferModel` example.
- **No compile recipe yet.** Getting from your `.pt` to a HEF is described, not scripted.

## Roadmap

**v0.2**, each step once the hardware or a Linux machine is available:

- A **tested compile recipe**: ONNX → Dataflow Compiler → HEF in WSL2/Ubuntu on x86, with
  the NMS config derived from the input size and the calibration set prepared with this
  repo's letterbox.
- A **verified `InferModel` example** for the installed HailoRT version.
- A **hardware run** of the doctor, `hefcheck` and both examples, recorded in the table
  above with versions and dates.
- A decision on **Trixie / HailoRT 4.23** support, checked against its source first.

## Development

Python 3.11 or newer. No Hailo software is needed to run the checks.

```bash
pip install -e ".[dev]"
```

```bash
ruff check . && ruff format --check . && mypy && pytest
```

The package needs numpy 1.x, because the `hailo_platform` that `apt` installs is built
against it. OpenCV is only needed to letterbox images. On the Pi it comes from `apt`;
elsewhere, use `pip install -e ".[opencv]"`.

## License

[MIT](LICENSE)

Built with Claude Code; design and review by me.
