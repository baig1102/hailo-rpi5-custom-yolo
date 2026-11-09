# 3. From your trained model to a HEF: an overview

> **Not yet a tested recipe.** This page explains the steps and what has to line up
> between them, so you know what to look for in Hailo's documentation. It hasn't been run
> end to end for this repo; a tested, scripted recipe is planned for v0.2 (see the
> README's roadmap). Commands and options belong to Hailo's tools, so take them from
> Hailo's documentation for your exact version.

## Two chains, two machines

| | Build chain | Run chain |
|---|---|---|
| Where | an x86 Linux machine (or WSL2) | the Raspberry Pi |
| Hailo software | Dataflow Compiler (DFC) | HailoRT (`hailo-all`) |
| Input | your trained model, exported to ONNX | the HEF |
| Output | a HEF | detections |

The DFC doesn't run on the Pi, and the Pi only needs the HEF. Get the DFC from the
[Hailo Developer Zone](https://hailo.ai/developer-zone/) (needs a Hailo account). This repo
doesn't redistribute it.

## The steps

1. **Export to ONNX** at a **fixed input size** (e.g. 640×640). That size is the model
   input the Pi will letterbox into.
2. **Parse:** the DFC reads the ONNX and builds its own model representation. For YOLO
   with built-in NMS, the graph is cut before the original post-processing, so that
   Hailo's NMS takes over.
3. **Optimise (quantise):** the model is converted to 8-bit using a **calibration set** of
   your own images. This step decides most of the accuracy on the chip.
4. **Compile** for the target architecture: **`hailo8l`** for the AI Kit / AI HAT+ 13 TOPS,
   `hailo8` for a Hailo-8. The result is the HEF.

[Hailo's Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) wraps these steps for
known model families, including YOLO, and is the usual starting point for a custom one.

## What has to line up (the mistakes this repo's tools catch)

- **Target architecture = your chip.** A Hailo-8L refuses a HEF compiled for `hailo8`.
  Check with `hailo-yolo-hefcheck --device-arch HAILO8L`.
- **NMS image size = input size.** The NMS config has its own image height and width.
  If they stay at e.g. 640 while the model input is 1280, every box comes out scaled
  wrong. `hefcheck` compares the two.
- **DFC release = HailoRT release on the Pi.** Hailo releases the DFC and HailoRT as a
  matching set. A HEF from a newer DFC than the Pi's HailoRT may fail to load or
  configure.
- **Calibration images = inference images.** Same camera or domain, and the same
  preprocessing as on the Pi: letterbox with grey padding, RGB order.
  `hailo_yolo.preprocess.letterbox_image` is the exact function the Pi uses, so you can
  prepare the calibration set with it.
- **NMS thresholds on purpose.** The score and IoU thresholds compiled into the HEF are
  the floor for everything the Pi sees. An IoU threshold of 0.7 is loose: overlapping
  boxes of one object survive.

## After compiling

Copy the HEF to the Pi, then check and run it:
[run your model](02-run-your-model.md).
