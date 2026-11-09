# Troubleshooting

Each entry is **symptom → cause → fix**, with its source:

- **"My setup"** means it happened on my Pi 5 + Hailo-8L with HailoRT 4.20.
- **"HailoRT 4.20"** points to the code in
  [hailo-ai/hailort](https://github.com/hailo-ai/hailort) at tag `v4.20.0`.

Start with `./scripts/doctor.sh`: it catches 1, 2 and parts of 6, 10 and 11 by itself.

## 1. "Board Name: Hailo-8", but it's a Hailo-8L

**Symptom.** `hailortcli fw-control identify` says `Board Name: Hailo-8` on an AI Kit or
AI HAT+ 13 TOPS. A HEF compiled for `hailo8` then fails to load.

**Cause.** The board name isn't the chip. `Device Architecture: HAILO8L` is. A Hailo-8L
only accepts HEFs compiled for `hailo8l`. A Hailo-8 accepts both, with a lower-performance
warning for `hailo8l` HEFs.

**Fix.** Compile for `hailo8l`. Check a HEF with
`hailo-yolo-hefcheck model.hef --device-arch HAILO8L`.

*Source:* my setup; HailoRT 4.20 `hailortcli/fw_control_command.cpp` lines 177–178 and
`libhailort/src/device_common/device_internal.cpp` line 716 (`is_hef_compatible`).

## 2. Version mismatch

**Symptom.** One of these:

- `Driver version (x.y.z) is different from library version (a.b.c)`
- `Unsupported firmware operation. Host: …, Device: …`
- `ImportError: _pyhailort version (…) does not match pyhailort version (…)`
- Other errors on load or configure that don't point anywhere useful.

**Cause.** HailoRT, the PCIe driver, the firmware and the Python bindings come from
different releases, e.g. after a partial `apt upgrade` or a driver update without a reboot.
The HEF must also come from the Dataflow Compiler release that matches HailoRT.

**Fix.** Run `./scripts/doctor.sh`: it names the part that differs. Install the matching
packages (`sudo apt install hailo-all`), reboot, and run the doctor again. To keep a
working set, hold the Hailo packages with `apt-mark hold` (Raspberry Pi's AI software page
shows how).

*Source:* my setup; HailoRT 4.20 `libhailort/src/vdma/driver/hailort_driver.cpp`
lines 231–241 (driver must equal the library), `libhailort/src/device_common/device.cpp`
lines 456–468 (firmware major.minor), `bindings/python/platform/hailo_platform/__init__.py`
lines 46–50.

## 3. Boxes are scaled wrong

**Symptom.** Boxes are consistently too small or too large, or shifted. It's tempting to
"fix" them with a factor measured by hand.

**Cause.** The model was compiled at one input size (e.g. 1280) while the NMS config
kept another image size (e.g. 640). The NMS normalises boxes with its own size.

**Fix.** Recompile with the NMS image size equal to the input size. No correction factor
in code. `hailo-yolo-hefcheck` reports this mismatch: it compares the input shape with
the NMS op's `Image height` and `Image width`.

*Source:* my setup; the fields are printed by HailoRT 4.20
`libhailort/src/net_flow/ops/yolov8_post_process.cpp` line 34.

## 4. Colours are swapped (Picamera2)

**Symptom.** Accuracy from the camera is worse than on photos, or saved frames have red
and blue swapped.

**Cause.** Picamera2's format names are the reverse of the array order: `"BGR888"`
arrays are R, G, B, and `"RGB888"` arrays are B, G, R. YOLO expects R, G, B, and
`cv2.imwrite` expects B, G, R.

**Fix.** Convert with `hailo_yolo.preprocess.picamera2_to_rgb(frame, fmt)` before
inference. With `"RGB888"`, the frame can go to `cv2.imwrite` as it is. With `"BGR888"`,
swap it before `cv2.imwrite` (`frame[:, :, ::-1]`), not before inference.

*Source:* my setup; Picamera2 manual, Appendix A "Pixel and image formats", Table 1.

## 5. Boxes are in the wrong place: x and y swapped

**Symptom.** Boxes appear mirrored along the diagonal, or are wrong on non-square images.

**Cause.** HailoRT's NMS boxes are `[y_min, x_min, y_max, x_max, score]`, normalised to
0–1: **y first**. There are also two output layouts, chosen by `tf_nms_format`.

**Fix.** Use `hailo_yolo.postprocess`, which reads both layouts in this order.

*Source:* HailoRT 4.20 `libhailort/include/hailo/hailort.h` lines 1379–1385
(`hailo_bbox_float32_t`) and `pyhailort.py` lines 868–875 (the two layouts).

## 6. `HAILO_OUT_OF_PHYSICAL_DEVICES`

**Symptom.** The app fails to start with this status, often only at boot or right after
a crash.

**Cause.** Another process holds the chip. Typical cases: two startup mechanisms
(autostart entry and service) racing at boot, or a restart loop starting the new process
before the crashed one has let go.

**Fix.** Start the app from one place only. `examples/hailo-app.service` waits 5 s before a
restart, and `HefEngine` retries for up to 15 s. See who holds the device with
`sudo fuser -v /dev/hailo0`.

*Source:* my setup. In HailoRT 4.20 this status has no exception class of its own: it
arrives as a plain `HailoRTException` (`pyhailort.py` line 161).

## 7. `couldn't connect to display ":0"` at boot

**Symptom.** Started at boot, the app crashes at once. Started by hand later, it works.

**Cause.** The autostart runs before the desktop is up, and the app opens a window.

**Fix.** A headless app (like `examples/infer_picamera2.py`) doesn't need a display.
An app with a window has to start after the desktop: from the desktop session's autostart
(one place only, see 6), or wait for the display before opening the window.

*Source:* my setup.

## 8. `HAILO_TIMEOUT` in the NMS post-process

**Symptom.** Inference runs, then stops with `HAILO_TIMEOUT`, often when the camera
changes mode or resolution.

**Cause.** A frame went into the pipeline and its output was never read: inference
overlapped a camera reconfiguration from another thread.

**Fix.** Never run inference and camera reconfiguration at the same time. Let one worker
own both, as `examples/infer_picamera2.py` does.

*Source:* my setup.

## 9. `HAILO_STREAM_ABORT` at shutdown

**Symptom.** Stopping the app (Ctrl+C, `systemctl stop`) prints `HAILO_STREAM_ABORT`.

**Cause.** The device was released while an inference was still running.

**Fix.** Stop the worker first, then close in a fixed order: pipeline, deactivate, release.
`HefEngine` closes in that order, and the camera example stops its worker on SIGINT and
SIGTERM before anything is released.

*Source:* my setup.

## 10. `HAILO_OPEN_FILE_FAILURE`

**Symptom.** The HEF can't be opened, usually only when started by a service.

**Cause.** The path is relative and the working directory differs (services don't start
in your project folder), or the path contains non-ASCII characters.

**Fix.** Use an absolute path and check that the file exists. `HefEngine` does both and
says `HEF not found: /full/path`. Keep model folders to plain ASCII names.

*Source:* my setup.

## 11. `HAILO_DRIVER_OPERATION_FAILED`

**Symptom.** Driver errors, typically after the app crashed or was killed.

**Cause.** The driver is in a bad state after an unclean stop.

**Fix.** Run `dmesg | grep hailo` (HailoRT's own message suggests this), then reboot. If it
keeps coming back with PCIe Gen 3 enabled, remove `dtparam=pciex1_gen=3` and test with
Gen 2.

*Source:* my setup; message from HailoRT 4.20 `pyhailort.py` line 153.

## 12. Accuracy is worse than on the PC

**Symptom.** Missed or extra detections compared with the original model.

**Cause.** Not the API. `InferVStreams` and `InferModel` give the same results with the
same HEF. Accuracy comes from three places:

- the calibration set used for quantisation;
- preprocessing that matches training (letterbox, RGB);
- the NMS thresholds compiled into the HEF.

**Fix.** Calibrate with images from your real camera and scene, preprocessed exactly like
inference. Check the colour order (4). Choose the thresholds on purpose: an IoU threshold
of 0.7 is loose and lets overlapping boxes of one object survive.

*Source:* my setup.

## 13. `InferModel` examples from the internet fail

**Symptom.** Copied `InferModel` code raises errors about missing methods or arguments.

**Cause.** Its Python API changed between HailoRT versions, and most examples online were
written for a different version than the one installed.

**Fix.** Check your version with `hailortcli --version` and use examples from exactly that
release. This repo uses the older `InferVStreams` API, which worked on my setup. A
verified `InferModel` example is planned for v0.2.

*Source:* my setup.
