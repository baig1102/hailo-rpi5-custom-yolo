# 2. Run your model

You have a working Pi ([setup](01-pi-setup.md)) and a HEF compiled from your own YOLO
model. This page covers checking the HEF, running it, and reading what comes back.

## What the HEF must look like

The code in this repo expects a HEF that meets three conditions:

- **Compiled for your chip.** A Hailo-8L only loads HEFs compiled for `hailo8l`. A Hailo-8
  also runs `hailo8l` HEFs, at lower performance. This rule is from HailoRT 4.20,
  `DeviceBase::is_hef_compatible`.
- **Built-in NMS.** The HEF has one NMS output, not the raw output layers. The boxes then
  come out already filtered and decoded.
- **NMS image size = input size.** The NMS is configured with an image height and width.
  If they differ from the model's input size, every box comes out scaled wrong (see
  [troubleshooting](troubleshooting.md#3-boxes-are-scaled-wrong)).

How to compile one is outlined in the [compile overview](03-compile-overview.md).

## Check the HEF first

On the Pi:

```bash
.venv/bin/hailo-yolo-hefcheck /home/pi/models/my_model.hef --device-arch HAILO8L
```

Take `--device-arch` from the doctor, or from the `Device Architecture` line of
`hailortcli fw-control identify`. Example output for a good HEF:

```
INFO     HEF compiled for HAILO8L matches the device.
INFO     NMS op YOLOV8: 3 classes, score threshold 0.25, IoU threshold 0.45.
INFO     Input custom_yolov8s/input_layer1 and the NMS image size are both 640x640.
```

To check a HEF away from the Pi, save the text there and check it anywhere:

```bash
hailortcli parse-hef my_model.hef > parse_hef.txt
hailo-yolo-hefcheck --from-file parse_hef.txt --device-arch HAILO8L
```

## Run it on one image

```bash
.venv/bin/python examples/infer_image.py /home/pi/models/my_model.hef photo.jpg out.jpg --threshold 0.5
```

This prints one line per detection and writes `out.jpg` with the boxes drawn.

## Use it from your own code

```python
from hailo_yolo.engine import HefEngine

with HefEngine("/home/pi/models/my_model.hef") as engine:
    detections = engine.detect(rgb_image, score_threshold=0.5)
```

- **`rgb_image`** is a uint8 array of any size in **R, G, B** order. Convert OpenCV
  images (`B, G, R`) with `bgr[:, :, ::-1]`, and Picamera2 frames with
  `hailo_yolo.preprocess.picamera2_to_rgb(frame, "RGB888")`. Picamera2's format names
  are the reverse of the array order.
- **`detect` handles the letterbox for you.** It resizes and pads like Ultralytics, runs
  the model, and maps the boxes back.
- **Each `Detection`** has `class_id` (the index in your training class list), `score`
  (0–1), and `x1, y1, x2, y2` in pixels of `rgb_image`.

Behind `detect`:

1. `letterbox_image` fits the image into the model input: aspect ratio kept, grey (114)
   padding.
2. `infer` sends that one frame through `InferVStreams` with `tf_nms_format=True`.
3. `to_detections` keeps scores ≥ the threshold, reads the boxes as
   `[y_min, x_min, y_max, x_max]` normalised to the input (**y comes first**), and undoes
   the letterbox.

## Thresholds: two of them

- The **NMS score threshold inside the HEF** is fixed at compile time; `hefcheck` shows
  it. HailoRT's NMS drops boxes below it before your code sees any output.
- **`score_threshold` in `detect`** filters what is left. It can only be *stricter* than
  the HEF's: setting it lower doesn't bring back boxes the HEF already dropped.

## Live camera and running at boot

- **`examples/infer_picamera2.py`** runs the camera live. One worker thread owns both the
  camera and the engine, so inference never overlaps a camera change or a shutdown.
- **`examples/hailo-app.service`** starts it at boot as a systemd user service. The file
  explains how to install it.

When something fails, see [troubleshooting](troubleshooting.md).
