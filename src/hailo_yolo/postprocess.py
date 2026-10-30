"""Turn the output of a HEF with built-in NMS into detections.

HailoRT returns NMS results in one of two layouts, picked by ``tf_nms_format`` when the
``InferVStreams`` pipeline is created. Both describe one box as five values in the order
``y_min, x_min, y_max, x_max, score``: y comes first. Coordinates are normalised to 0-1.

Sources, all in hailo-ai/hailort at tag v4.20.0:

- Box field order: ``hailo_bbox_float32_t``,
  hailort/libhailort/include/hailo/hailort.h lines 1379-1385.
- Both layouts: ``InferVStreams`` docstring,
  hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py
  lines 868-875, and the converters in the same file (``output_raw_buffer_to_nms_tf_format``
  from line 1245, ``output_raw_buffer_to_nms_format_single_frame`` from line 1275).
- Empty slots in the TF layout: filled with a quantised zero box that is dequantised for
  float32 outputs (same file, lines 3872-3875). With a non-zero zero point that is not
  exactly 0, so empty slots are dropped by score, never by checking for zeros.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hailo_yolo.preprocess import Letterbox, boxes_to_source

# Values per box: y_min, x_min, y_max, x_max, score.
BBOX_PARAMS = 5


@dataclass(frozen=True)
class NmsDetections:
    """NMS results of one frame, in HailoRT's own coordinates.

    ``boxes`` rows are ``[y_min, x_min, y_max, x_max]``, normalised to the model input.
    """

    class_ids: npt.NDArray[np.int64]
    boxes: npt.NDArray[np.float64]
    scores: npt.NDArray[np.float64]


def from_hailo_format(per_class: Sequence[npt.ArrayLike]) -> NmsDetections:
    """Read one frame of the default layout (``tf_nms_format=False``).

    ``per_class`` holds one array of shape ``(detections, 5)`` per class; a class without
    detections has an array of shape ``(0, 5)``. ``InferVStreams.infer`` returns a list of
    these per output, one entry per frame in the batch.
    """
    arrays = [np.asarray(dets, dtype=np.float64) for dets in per_class]
    for class_id, arr in enumerate(arrays):
        if arr.ndim != 2 or arr.shape[1] != BBOX_PARAMS:
            raise ValueError(f"class {class_id}: expected shape (detections, 5), got {arr.shape}")
    class_ids = np.repeat(np.arange(len(arrays), dtype=np.int64), [len(a) for a in arrays])
    rows = np.concatenate(arrays) if arrays else np.zeros((0, BBOX_PARAMS))
    return _split(class_ids, rows)


def from_tf_format(frame: npt.ArrayLike) -> NmsDetections:
    """Read one frame of the TensorFlow layout (``tf_nms_format=True``).

    ``frame`` has shape ``(classes, 5, max_boxes_per_class)``: the box values run along the
    middle axis. Every slot is returned, including the empty padding slots; drop those with
    a score threshold.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != BBOX_PARAMS:
        raise ValueError(f"TF NMS frame must have shape (classes, 5, max_boxes), got {arr.shape}")
    classes, _, max_boxes = arr.shape
    # (classes, 5, max_boxes) -> (classes, max_boxes, 5) -> one row per slot
    rows = arr.transpose(0, 2, 1).reshape(-1, BBOX_PARAMS)
    class_ids = np.repeat(np.arange(classes, dtype=np.int64), max_boxes)
    return _split(class_ids, rows)


def _split(class_ids: npt.NDArray[np.int64], rows: npt.NDArray[np.float64]) -> NmsDetections:
    return NmsDetections(class_ids=class_ids, boxes=rows[:, :4], scores=rows[:, 4])


@dataclass(frozen=True)
class Detection:
    """One detection in source-image pixels, ``(x1, y1)`` top left, ``(x2, y2)`` bottom right."""

    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def to_detections(nms: NmsDetections, lb: Letterbox, *, score_threshold: float) -> list[Detection]:
    """Keep detections with ``score >= score_threshold`` and map them onto the source image.

    The threshold has no default on purpose: it is a decision about your model, and it is
    also what drops the empty padding slots of the TF layout, so it must be above 0.
    Detections come back sorted by score, highest first.
    """
    if not 0 < score_threshold <= 1:
        raise ValueError(f"score_threshold must be in (0, 1], got {score_threshold}")

    keep = nms.scores >= score_threshold
    y_min, x_min, y_max, x_max = nms.boxes[keep].T
    # Normalised [y_min, x_min, y_max, x_max] -> model-input pixels [x1, y1, x2, y2]
    input_xyxy = np.stack(
        [x_min * lb.dst_width, y_min * lb.dst_height, x_max * lb.dst_width, y_max * lb.dst_height],
        axis=1,
    )
    boxes = boxes_to_source(input_xyxy, lb)
    class_ids = nms.class_ids[keep]
    scores = nms.scores[keep]

    order = np.argsort(-scores, kind="stable")
    return [
        Detection(
            class_id=int(class_ids[i]),
            score=float(scores[i]),
            x1=float(boxes[i, 0]),
            y1=float(boxes[i, 1]),
            x2=float(boxes[i, 2]),
            y2=float(boxes[i, 3]),
        )
        for i in order
    ]
