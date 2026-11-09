"""Live detection from the Pi camera: one worker thread owns both the camera and the engine.

    python examples/infer_picamera2.py model.hef --size 1280x720 --threshold 0.5

Why one owner: a frame sent to the Hailo while another thread reconfigures the camera, or
closes the engine, was how ``HAILO_TIMEOUT`` and ``HAILO_STREAM_ABORT`` happened on my setup.
Here the worker is the only code that touches either; the main thread only sets a stop flag
on Ctrl+C or on SIGTERM (what systemd sends on ``systemctl stop``).

Needs Picamera2 and OpenCV from apt on the Pi (``python3-picamera2``, ``python3-opencv``).
"""

import argparse
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import numpy.typing as npt

from hailo_yolo.engine import HefEngine
from hailo_yolo.postprocess import Detection
from hailo_yolo.preprocess import picamera2_to_rgb

# Arrays in this format come out as B, G, R (Picamera2 manual, Appendix A, Table 1): ready for
# OpenCV drawing, and picamera2_to_rgb turns them into the R, G, B the model expects.
CAMERA_FORMAT = "RGB888"

OnFrame = Callable[[npt.NDArray[np.uint8], list[Detection]], None]


@contextmanager
def open_camera(width: int, height: int) -> Iterator[Any]:
    """Start the Pi camera; stop and close it on exit. Picamera2 is imported only here."""
    from picamera2 import Picamera2

    camera = Picamera2()
    try:
        main = {"format": CAMERA_FORMAT, "size": (width, height)}
        camera.configure(camera.create_preview_configuration(main=main))
        camera.start()
        yield camera
    finally:
        camera.close()


def run(
    camera: Any, engine: HefEngine, stop: threading.Event, on_frame: OnFrame, threshold: float
) -> int:
    """Capture, detect and report frames until ``stop`` is set. Returns the frame count."""
    frames = 0
    while not stop.is_set():
        frame = np.asarray(camera.capture_array("main"), dtype=np.uint8)
        detections = engine.detect(
            picamera2_to_rgb(frame, CAMERA_FORMAT), score_threshold=threshold
        )
        on_frame(frame, detections)
        frames += 1
    return frames


def worker(
    hef: Path,
    size: tuple[int, int],
    threshold: float,
    stop: threading.Event,
    on_frame: OnFrame,
    camera_factory: Callable[[int, int], Any] = open_camera,
) -> int:
    """Open the camera, then the engine; on exit, close the engine first, then the camera."""
    with camera_factory(*size) as camera, HefEngine(hef) as engine:
        return run(camera, engine, stop, on_frame, threshold)


def print_rate(every: int = 30) -> OnFrame:
    """An ``on_frame`` callback that prints frames per second and the latest detections."""
    state = {"count": 0, "since": time.monotonic()}

    def on_frame(frame: npt.NDArray[np.uint8], detections: list[Detection]) -> None:
        state["count"] += 1
        if state["count"] % every == 0:
            now = time.monotonic()
            fps = every / (now - state["since"])
            state["since"] = now
            found = ", ".join(f"{d.class_id}:{d.score:.2f}" for d in detections) or "nothing"
            print(f"{fps:5.1f} fps  {found}", flush=True)

    return on_frame


def parse_size(text: str) -> tuple[int, int]:
    width, _, height = text.partition("x")
    return int(width), int(height)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hef", type=Path)
    parser.add_argument("--size", type=parse_size, default=(1280, 720), help="camera WxH")
    parser.add_argument("--threshold", type=float, default=0.5, help="score threshold, 0..1")
    args = parser.parse_args(argv)

    stop = threading.Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    errors: list[BaseException] = []

    def target() -> None:
        try:
            worker(args.hef, args.size, args.threshold, stop, print_rate())
        except BaseException as e:  # reported by the main thread below
            errors.append(e)

    thread = threading.Thread(target=target, name="camera-and-inference")
    thread.start()
    while thread.is_alive():
        thread.join(timeout=0.5)  # short waits keep the main thread free for signals
    if errors:
        print(f"stopped by an error: {errors[0]!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
