# hailo-rpi5-custom-yolo

[![CI](https://github.com/baig1102/hailo-rpi5-custom-yolo/actions/workflows/ci.yml/badge.svg)](https://github.com/baig1102/hailo-rpi5-custom-yolo/actions/workflows/ci.yml)

Set up a Hailo-8L on a Raspberry Pi 5 and run **your own** custom-trained YOLO
model, with small tools that catch the common mistakes before they cost you a day.

Written from my own experience deploying a custom model on a Pi 5 + Hailo-8L.
Contains no employer code or data.

> **Interim README. Work in progress.** The project skeleton and CI are in place;
> the tools themselves are still to come. The plan is below; the
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

## Status

| # | Milestone | Status |
|---|---|---|
| 1 | Project skeleton, CI ([#1](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/1)) | ✅ done |
| 2 | Letterbox preprocessing, NMS postprocessing ([#2](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/2)) | ⏳ planned |
| 3 | `hefcheck`: architecture, input vs NMS size ([#3](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/3)) | ⏳ planned |
| 4 | `doctor`: version, device and import checks ([#4](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/4)) | ⏳ planned |
| 5 | Inference engine, `InferVStreams` path ([#5](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/5)) | ⏳ planned |
| 6 | Examples, setup guides, troubleshooting ([#6](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/6)) | ⏳ planned |
| 7 | Final README, `v0.1.0` ([#7](https://github.com/baig1102/hailo-rpi5-custom-yolo/issues/7)) | ⏳ planned |

## Development

Python 3.11 or newer. No Hailo software is needed to run the checks.

```bash
pip install -e ".[dev]"
```

```bash
ruff check . && ruff format --check . && mypy && pytest
```

## License

[MIT](LICENSE)

Built with Claude Code; design and review by me.
