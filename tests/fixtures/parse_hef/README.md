# Synthetic `hailortcli parse-hef` outputs

Written by hand for the tests. They are **not** captured from a device or from any real
model. Each one follows the exact text format built by HailoRT v4.20.0
(hailo-ai/hailort, `hailort/libhailort/src/hef/hef.cpp`, `Hef::Impl::get_description` from
line 3906, and the NMS op descriptions in `src/net_flow/ops/`), with generic names.

| File | What it shows |
|---|---|
| `yolov8_hailo8l_640.txt` | Correct HEF: Hailo-8L, input 640x640, NMS image size 640x640 |
| `yolov8_hailo8_640.txt` | Compiled for Hailo-8: won't load on a Hailo-8L |
| `yolov8_input1280_nms640.txt` | Input 1280x1280 but NMS configured for 640x640: boxes scale wrong |
| `yolov8_no_nms.txt` | No built-in NMS: raw output layers only |
