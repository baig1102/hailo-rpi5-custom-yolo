"""Read ``hailortcli parse-hef`` output and check a HEF before running it.

The text format comes from HailoRT v4.20.0 (hailo-ai/hailort):

- ``hailort/libhailort/src/hef/hef.cpp``: ``Hef::Impl::get_description`` (line 3906) writes
  the architecture line, the network group and the vstreams; ``get_shape_str`` (line 105)
  the shapes; ``get_post_processes_infos_description`` (line 3869) splits the NMS op
  description at ", " into one line per field.
- ``hailort/libhailort/src/net_flow/ops/nms_post_process.cpp``: the NMS fields (line 86);
  ``yolov8_post_process.cpp`` (line 34) adds ``Image height`` and ``Image width``.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VStream:
    """One input or output vstream, e.g. ``Input  net/input_layer1 UINT8, NHWC(640x640x3)``."""

    direction: str  # "input" or "output"
    name: str
    format_type: str  # e.g. UINT8, FLOAT32
    format_order: str  # e.g. NHWC, HAILO NMS BY CLASS
    shape: tuple[int, ...] | None  # (H, W, C) for NHWC; None for NMS outputs
    nms_classes: int | None = None
    nms_max_bboxes_per_class: int | None = None


@dataclass(frozen=True)
class NmsOp:
    """The built-in NMS post-process op, if the HEF has one."""

    op_type: str  # e.g. YOLOV8, YOLOV5
    score_threshold: float
    iou_threshold: float
    classes: int
    image_height: int | None  # only YOLO ops print these
    image_width: int | None


@dataclass(frozen=True)
class HefInfo:
    architecture: str  # e.g. HAILO8L
    network_groups: tuple[str, ...]
    inputs: tuple[VStream, ...]
    outputs: tuple[VStream, ...]
    nms: NmsOp | None


_ARCH = re.compile(r"^Architecture HEF was compiled for: (?P<arch>\S+)", re.MULTILINE)
_GROUP = re.compile(r"^Network group name: (?P<name>[^,]+),", re.MULTILINE)
# e.g. "Input  net/input_layer1 UINT8, NHWC(640x640x3)"
_VSTREAM = re.compile(
    r"^\s*(?P<dir>Input|Output)\s+(?P<name>\S+) (?P<type>\w+), "
    r"(?P<order>[A-Z0-9 ]+?)\((?P<args>.*)\)\s*$",
    re.MULTILINE,
)
_NMS_ARGS = re.compile(
    r"number of classes: (?P<classes>\d+), maximum bounding boxes per class: (?P<max>\d+)"
)


def parse_hef_output(text: str) -> HefInfo:
    """Parse the text that ``hailortcli parse-hef <file.hef>`` prints."""
    arch = _ARCH.search(text)
    if arch is None:
        raise ValueError("no 'Architecture HEF was compiled for' line: is this parse-hef output?")

    inputs: list[VStream] = []
    outputs: list[VStream] = []
    for m in _VSTREAM.finditer(text):
        vstream = _parse_vstream(m)
        (inputs if vstream.direction == "input" else outputs).append(vstream)

    return HefInfo(
        architecture=arch["arch"],
        network_groups=tuple(m["name"] for m in _GROUP.finditer(text)),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        nms=_parse_nms_op(text),
    )


def _parse_vstream(m: re.Match[str]) -> VStream:
    order, args = m["order"].strip(), m["args"]
    shape: tuple[int, ...] | None = None
    classes = max_bboxes = None
    if order.startswith("HAILO NMS"):
        nms = _NMS_ARGS.search(args)
        if nms is not None:
            classes, max_bboxes = int(nms["classes"]), int(nms["max"])
    else:
        shape = tuple(int(v) for v in args.split("x"))
    return VStream(
        direction=m["dir"].lower(),
        name=m["name"],
        format_type=m["type"],
        format_order=order,
        shape=shape,
        nms_classes=classes,
        nms_max_bboxes_per_class=max_bboxes,
    )


def _parse_nms_op(text: str) -> NmsOp | None:
    _, found, block = text.partition("Operation:")
    if not found:
        return None
    # One field per line: "Op YOLOV8", then "Key: value" lines.
    fields: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("Op "):
            fields["Op"] = line[3:]
        elif ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    if "Score threshold" not in fields:
        return None  # an op without NMS, e.g. softmax or argmax
    height, width = fields.get("Image height"), fields.get("Image width")
    return NmsOp(
        op_type=fields["Op"],
        score_threshold=float(fields["Score threshold"]),
        iou_threshold=float(fields["IoU threshold"]),
        classes=int(fields["Classes"]),
        image_height=int(height) if height is not None else None,
        image_width=int(width) if width is not None else None,
    )


@dataclass(frozen=True)
class Finding:
    level: str  # "error": won't load, "warning": loads but results are wrong or slow, "info"
    message: str


# HEF architectures each device accepts: DeviceBase::is_hef_compatible,
# hailort/libhailort/src/device_common/device_internal.cpp line 716 (v4.20.0). A Hailo-8 also
# runs Hailo-8L HEFs, with a "lower performance" warning (same file, line 616).
_DEVICE_ACCEPTS = {"HAILO8": {"HAILO8", "HAILO8L"}, "HAILO8L": {"HAILO8L"}}


def check_hef(info: HefInfo, device_arch: str | None = None) -> list[Finding]:
    """Check a parsed HEF, optionally against the device's architecture (e.g. ``HAILO8L``).

    ``device_arch`` is the ``Device Architecture`` line of ``hailortcli fw-control identify``,
    not ``Board Name``: a Hailo-8L reports ``Board Name: Hailo-8``.
    """
    return [*_check_architecture(info, device_arch), *_check_nms(info)]


def _check_architecture(info: HefInfo, device_arch: str | None) -> list[Finding]:
    hef = info.architecture
    if device_arch is None:
        return [
            Finding(
                "info",
                f"HEF compiled for {hef}. Compare it with 'Device Architecture' from "
                "'hailortcli fw-control identify', not with 'Board Name'.",
            )
        ]
    device = device_arch.upper()
    accepts = _DEVICE_ACCEPTS.get(device)
    if accepts is None:
        return [Finding("info", f"HEF compiled for {hef}; device {device} is not checked here.")]
    if hef not in accepts:
        return [
            Finding(
                "error",
                f"HEF compiled for {hef} won't load on a {device} device. "
                f"Compile it for {device.lower()}.",
            )
        ]
    if hef != device:
        return [
            Finding(
                "warning",
                f"HEF compiled for {hef} runs on a {device} device, but with lower performance. "
                f"Compile it for {device.lower()} to use the whole chip.",
            )
        ]
    return [Finding("info", f"HEF compiled for {hef} matches the device.")]


def _check_nms(info: HefInfo) -> list[Finding]:
    nms = info.nms
    if nms is None:
        return [
            Finding(
                "warning",
                "No built-in NMS op: the outputs are raw layers. hailo_yolo.postprocess expects "
                "a HEF compiled with NMS.",
            )
        ]

    findings = [
        Finding(
            "info",
            f"NMS op {nms.op_type}: {nms.classes} classes, score threshold "
            f"{nms.score_threshold:g}, IoU threshold {nms.iou_threshold:g}.",
        )
    ]
    if nms.image_height is None or nms.image_width is None:
        findings.append(Finding("info", "The NMS op reports no image size; not checked."))
        return findings

    nms_hw = f"{nms.image_height}x{nms.image_width}"
    for inp in info.inputs:
        if inp.shape is None or len(inp.shape) != 3:
            continue
        input_hw = f"{inp.shape[0]}x{inp.shape[1]}"
        if input_hw != nms_hw:
            findings.append(
                Finding(
                    "warning",
                    f"Input {inp.name} is {input_hw} (HxW) but the NMS is configured for "
                    f"{nms_hw}: boxes will come out scaled wrong. Set the NMS image size to the "
                    "input size when compiling.",
                )
            )
        else:
            findings.append(
                Finding("info", f"Input {inp.name} and the NMS image size are both {input_hw}.")
            )
    return findings
