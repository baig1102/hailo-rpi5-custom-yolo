import importlib
import importlib.util
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from fake_hailo_platform import FakeHailo

from hailo_yolo.postprocess import Detection

EXAMPLES = Path(__file__).parents[1] / "examples"


def load_example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_hailo(monkeypatch: pytest.MonkeyPatch) -> FakeHailo:
    """Make ``import hailo_platform`` inside the engine return the fake."""
    fake = FakeHailo()
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        return fake.module() if name == "hailo_platform" else real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return fake


def test_infer_image_draws_the_detection_and_saves(
    tmp_path: Path, fake_hailo: FakeHailo, capsys: pytest.CaptureFixture[str]
) -> None:
    hef, image, output = tmp_path / "model.hef", tmp_path / "in.png", tmp_path / "out.png"
    hef.write_bytes(b"")
    cv2.imwrite(str(image), np.zeros((640, 640, 3), dtype=np.uint8))

    code = load_example("infer_image").main([str(hef), str(image), str(output)])

    assert code == 0
    # The fake's box: x 160..320, y 160..480 on a 640 x 640 image.
    assert "class 1  score 0.90  box 160,160,320,480" in capsys.readouterr().out
    drawn = cv2.imread(str(output))
    assert drawn[300, 160].tolist() == [0, 255, 0]  # left edge of the box, green in BGR
    assert drawn[300, 240].tolist() == [0, 0, 0]  # inside the box: untouched
    assert fake_hailo.log[-1] == "VDevice.release"


def test_infer_image_missing_image_exits_2(tmp_path: Path, fake_hailo: FakeHailo) -> None:
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"")
    code = load_example("infer_image").main([str(hef), str(tmp_path / "no.png"), "out.png"])
    assert code == 2
    assert fake_hailo.log == []  # the device was never opened


class FakeCamera:
    """Stands in for Picamera2: RGB888 frames are B, G, R arrays of the configured size."""

    def __init__(self, log: list[str], width: int, height: int) -> None:
        self.log, self.shape = log, (height, width, 3)

    def capture_array(self, name: str) -> npt.NDArray[np.uint8]:
        self.log.append("capture")
        return np.zeros(self.shape, dtype=np.uint8)


def fake_camera_factory(log: list[str]) -> object:
    @contextmanager
    def factory(width: int, height: int) -> Iterator[FakeCamera]:
        log.append("camera.open")
        try:
            yield FakeCamera(log, width, height)
        finally:
            log.append("camera.close")

    return factory


def test_worker_owns_camera_and_engine_and_closes_in_order(
    tmp_path: Path, fake_hailo: FakeHailo
) -> None:
    example = load_example("infer_picamera2")
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"")
    stop = threading.Event()
    log = fake_hailo.log  # camera and Hailo calls in one log, so their order is visible
    seen: list[int] = []

    def on_frame(frame: npt.NDArray[np.uint8], detections: list[Detection]) -> None:
        seen.append(len(detections))
        if len(seen) == 3:
            stop.set()

    frames = example.worker(
        hef, (1280, 720), 0.5, stop, on_frame, camera_factory=fake_camera_factory(log)
    )

    assert frames == 3
    assert seen == [1, 1, 1]
    assert log[:2] == ["camera.open", "HEF"]  # camera first, then the engine
    assert [e for e in log if e in ("capture", "infer")] == ["capture", "infer"] * 3
    # Engine shut down completely before the camera closes.
    assert log[-4:] == ["pipeline.exit", "activation.exit", "VDevice.release", "camera.close"]


def test_worker_closes_the_camera_when_the_engine_fails(
    tmp_path: Path, fake_hailo: FakeHailo
) -> None:
    example = load_example("infer_picamera2")
    camera_log: list[str] = []
    with pytest.raises(FileNotFoundError):
        example.worker(
            tmp_path / "missing.hef",
            (640, 480),
            0.5,
            threading.Event(),
            lambda frame, detections: None,
            camera_factory=fake_camera_factory(camera_log),
        )
    assert camera_log == ["camera.open", "camera.close"]


def test_parse_size() -> None:
    assert load_example("infer_picamera2").parse_size("1280x720") == (1280, 720)
