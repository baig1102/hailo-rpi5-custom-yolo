"""Run a HEF on one image and save a copy with the detections drawn.

    python examples/infer_image.py model.hef photo.jpg photo_boxes.jpg --threshold 0.5

Needs OpenCV (on the Pi: ``sudo apt install python3-opencv``).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hailo_yolo.engine import HefEngine
from hailo_yolo.postprocess import Detection


def draw(bgr: npt.NDArray[np.uint8], detections: list[Detection], labels: list[str]) -> None:
    """Draw boxes and labels in place on a BGR image (OpenCV's channel order)."""
    for d in detections:
        name = labels[d.class_id] if d.class_id < len(labels) else str(d.class_id)
        top_left, bottom_right = (round(d.x1), round(d.y1)), (round(d.x2), round(d.y2))
        cv2.rectangle(bgr, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"{name} {d.score:.2f}",
            (top_left[0], max(top_left[1] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hef", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5, help="score threshold, 0..1")
    parser.add_argument("--labels", type=Path, help="text file, one class name per line")
    args = parser.parse_args(argv)

    loaded = cv2.imread(str(args.image))  # OpenCV loads B, G, R
    if loaded is None:
        print(f"can't read image {args.image}", file=sys.stderr)
        return 2
    bgr = np.asarray(loaded, dtype=np.uint8)  # already uint8; states it for the type checker
    labels = args.labels.read_text().splitlines() if args.labels else []

    rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # YOLO expects R, G, B
    with HefEngine(args.hef) as engine:
        detections = engine.detect(rgb, score_threshold=args.threshold)

    for d in detections:
        box = f"{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}"
        print(f"class {d.class_id}  score {d.score:.2f}  box {box}")
    draw(bgr, detections, labels)
    if not cv2.imwrite(str(args.output), bgr):
        print(f"can't write {args.output}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
