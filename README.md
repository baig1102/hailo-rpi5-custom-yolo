# hailo-rpi5-custom-yolo

[![CI](https://github.com/baig1102/hailo-rpi5-custom-yolo/actions/workflows/ci.yml/badge.svg)](https://github.com/baig1102/hailo-rpi5-custom-yolo/actions/workflows/ci.yml)

Set up a Hailo-8L on a Raspberry Pi 5 and run **your own** custom-trained YOLO
model, with small tools that catch the common mistakes before they cost you a day.

Written from my own experience deploying a custom model on a Pi 5 + Hailo-8L.
Contains no employer code or data.

> **Interim README. Work in progress.** The tools, examples and guides are in place, and
> the code is unit-tested; the final README comes next. The plan is below; the
> final README (quick start, verification status, limitations) comes with v0.1.0.
> Nothing in this repo has been validated on hardware yet.

## The problem

Hailo's examples run Hailo's own models well. Running a model you trained
yourself is where it gets hard, and the docs say little about it:

- `hailortcli` reports `Board Name: Hailo-8` on a Hailo-8L. What decides whether
  a HEF loads is the *device architecture*, not the board name.
- HailoRT, the PCIe driver, the firmware and the Python bindings have to match,
  and the HEF must come from the matching compiler release. A mismatch gives
  errors that don't point at the cause.
- If the model's input size and the image size in the built-in NMS config
  differ, every box comes out scaled wrong, and it's tempting to "fix" that with
  a hand-measured factor.
- Picamera2's colour format names are the reverse of the array order you get,
  and the NMS output puts y before x.

This repo turns those lessons into a guide plus a few checks you can run on the Pi.

## Planned contents

- **`doctor`:** checks installed versions, the device and its architecture, and
  the Python import, and says what's wrong in plain words.
- **`hefcheck`:** reads `hailortcli parse-hef` output and warns about a HEF built
  for the wrong architecture or with an NMS size that doesn't match the input.
- **Pre- and post-processing:** letterbox and its inverse, NMS output to pixel
  boxes in the original image.
- **A small inference engine** on the `InferVStreams` API, with a clean
  start-up and shutdown.
- **Docs:** Pi setup, running your model, a compile overview and a
  troubleshooting page written as symptom → cause → fix.

## Guides

1. [Set up the Pi](docs/01-pi-setup.md): install, verify, run the doctor
2. [Run your model](docs/02-run-your-model.md): check the HEF, run it, read the output
3. [From your model to a HEF](docs/03-compile-overview.md): overview only, not yet a tested recipe
4. [Troubleshooting](docs/troubleshooting.md): 13 problems, each as symptom → cause → fix

Examples: [`infer_image.py`](examples/infer_image.py) (one image),
[`infer_picamera2.py`](examples/infer_picamera2.py) (live camera) and
[`hailo-app.service`](examples/hailo-app.service) (start at boot).

All logic is plain Python and unit-tested in CI without Hailo hardware.

## The setup this is based on

This is the setup I ran my own model on. The tools in this repo haven't been
run on it yet.

| Layer | Version |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1 |
| OS | Raspberry Pi OS Bookworm (Debian 12), 64-bit, kernel 6.12 |
| Accelerator | Hailo-8L (reports `Device Architecture: HAILO8L`) |
| `hailo-all` / HailoRT / driver / firmware | 4.20.0 |
| TAPPAS core | 3.31.0 |
| Python | 3.11.2 (system), numpy 1.26.4 |
| Camera stack | Picamera2 0.3.31 |

Hailo's software isn't included here. Get it from Raspberry Pi's and Hailo's
official sources.

Raspberry Pi's AI documentation now targets Raspberry Pi OS Trixie, which ships
HailoRT 4.23. The doctor knows Trixie's package names, but nothing here has been
tested against 4.23.

## Status

| # | Milestone | Status |
|---|---|---|
| 1 | Project skeleton, CI ([#1](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/1)) | ✅ done |
| 2 | Letterbox preprocessing, NMS postprocessing ([#2](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/2)) | ✅ done |
| 3 | `hefcheck`: architecture, input vs NMS size ([#3](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/3)) | ✅ done |
| 4 | `doctor`: version, device and import checks ([#4](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/4)) | ✅ done |
| 5 | Inference engine, `InferVStreams` path ([#5](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/5)) | ✅ done |
| 6 | Examples, setup guides, troubleshooting ([#6](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/6)) | ✅ done |
| 7 | Final README, `v0.1.0` ([#7](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/7)) | ⏳ planned |

## Development

Python 3.11 or newer. No Hailo software is needed to run the checks.

```bash
pip install -e ".[dev]"
```

```bash
ruff check . && ruff format --check . && mypy && pytest
```

The package needs numpy 1.x: the `hailo_platform` that `apt` installs is built
against it. OpenCV is only needed to letterbox images. On the Pi, use Bookworm's
`python3-opencv` from `apt`; elsewhere, `pip install -e ".[opencv]"`.

## Run a HEF

```python
from hailo_yolo.engine import HefEngine

with HefEngine("/home/pi/models/my_model.hef") as engine:
    detections = engine.detect(rgb_image, score_threshold=0.5)  # any size, RGB, uint8
for d in detections:
    print(d.class_id, round(d.score, 2), d.x1, d.y1, d.x2, d.y2)  # pixels on rgb_image
```

The engine uses the legacy `InferVStreams` API, the one that worked on my setup. It
checks the HEF path, retries while another process holds the chip, and always shuts
down in the same order. Let one thread own it. Not run on hardware yet; the tests use
a fake `hailo_platform`.

## Check the Pi

From a clone of this repo, with the Pi's system Python (no install needed):

```bash
./scripts/doctor.sh
```

It checks that HailoRT, the PCIe driver (installed and loaded), the firmware (installed
and running) and the Python bindings all have the same version. It also checks that a
device answers, shows its architecture, and checks that `hailo_platform` imports in this
Python. Each problem comes with what to do. Exit code 0 means no problems, 1 means a
warning or error was found.

## Check a HEF

On the Pi (runs `hailortcli parse-hef` for you):

```bash
hailo-yolo-hefcheck /home/pi/models/my_model.hef --device-arch HAILO8L
```

Anywhere else, on saved output (`hailortcli parse-hef my_model.hef > parse_hef.txt`):

```bash
hailo-yolo-hefcheck --from-file parse_hef.txt --device-arch HAILO8L
```

Take `--device-arch` from the `Device Architecture` line of
`hailortcli fw-control identify`, not from `Board Name`. Exit code 0 means no
problems, 1 means a warning or error was found, and 2 means the check couldn't run.

## License

[MIT](LICENSE)

Built with Claude Code; design and review by me.
