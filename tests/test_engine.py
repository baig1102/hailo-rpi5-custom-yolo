import importlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from fake_hailo_platform import FakeHailo, HailoRTException

from hailo_yolo.engine import HefEngine

OPEN_ORDER = [
    "HEF",
    "VDevice",
    "ConfigureParams(PCIe)",
    "configure",
    "InputVStreamParams(UINT8)",
    "OutputVStreamParams(FLOAT32)",
    "activate",
    "activation.enter",
    "InferVStreams(tf_nms_format=True)",
    "pipeline.enter",
]
CLOSE_ORDER = ["pipeline.exit", "activation.exit", "VDevice.release"]


@pytest.fixture
def hef(tmp_path: Path) -> Path:
    path = tmp_path / "model.hef"
    path.write_bytes(b"")  # only its existence matters to the fake
    return path


def engine(hef: Path, fake: FakeHailo) -> HefEngine:
    return HefEngine(hef, platform_module=fake.module())  # type: ignore[arg-type]


def test_opens_and_closes_in_a_fixed_order(hef: Path) -> None:
    fake = FakeHailo()
    with engine(hef, fake) as e:
        assert fake.log == OPEN_ORDER
        assert e.input_name == "net/input_layer1"
        assert e.input_shape == (640, 640, 3)
    assert fake.log == OPEN_ORDER + CLOSE_ORDER


@pytest.mark.parametrize(
    ("fail_on", "opened"),
    [
        ("configure", ["VDevice.release"]),
        ("activation.enter", ["VDevice.release"]),
        ("pipeline.enter", ["activation.exit", "VDevice.release"]),
    ],
)
def test_failure_while_opening_closes_what_was_opened(
    hef: Path, fail_on: str, opened: list[str]
) -> None:
    fake = FakeHailo(fail_on=fail_on)
    with pytest.raises(HailoRTException, match=fail_on):
        engine(hef, fake).open()
    assert fake.log[fake.log.index(fail_on) + 1 :] == opened


def test_exception_inside_the_with_block_still_closes_in_order(hef: Path) -> None:
    fake = FakeHailo()
    with pytest.raises(KeyboardInterrupt), engine(hef, fake):
        raise KeyboardInterrupt
    assert fake.log[-3:] == CLOSE_ORDER


def test_close_twice_is_safe(hef: Path) -> None:
    fake = FakeHailo()
    e = engine(hef, fake).open()
    e.close()
    e.close()
    assert fake.log.count("VDevice.release") == 1


def test_opening_twice_is_refused(hef: Path) -> None:
    with engine(hef, FakeHailo()) as e, pytest.raises(RuntimeError, match="already open"):
        e.open()


def test_missing_hef_is_reported_before_hailort_is_touched(tmp_path: Path) -> None:
    fake = FakeHailo()
    with pytest.raises(FileNotFoundError, match="HEF not found"):
        engine(tmp_path / "missing.hef", fake).open()
    assert fake.log == []


def test_relative_path_becomes_absolute(hef: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(hef.parent)
    e = HefEngine("model.hef")
    assert e.hef_path == hef.resolve()
    assert e.hef_path.is_absolute()


def test_hef_with_two_outputs_is_refused(hef: Path) -> None:
    fake = FakeHailo()
    module = fake.module()
    single = module.HEF.get_output_vstream_infos
    module.HEF.get_output_vstream_infos = lambda self: single(self) * 2
    with pytest.raises(ValueError, match="1 input and 1 NMS output"):
        HefEngine(hef, platform_module=module).open()  # type: ignore[arg-type]
    assert fake.log[-1] == "VDevice.release"


def test_hailo_platform_is_imported_on_open_not_before(
    hef: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeHailo()
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return fake.module()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    e = HefEngine(hef)
    assert imported == []
    with e:
        assert imported == ["hailo_platform"]


def test_busy_device_is_retried_with_doubling_waits(hef: Path) -> None:
    fake = FakeHailo(busy_attempts=3)
    waits: list[float] = []
    e = HefEngine(hef, sleep=waits.append, platform_module=fake.module())  # type: ignore[arg-type]
    with e:
        assert waits == [1.0, 2.0, 4.0]
        assert fake.log.count("VDevice busy") == 3
        assert fake.log.count("VDevice") == 1


def test_busy_device_gives_up_after_the_last_attempt(hef: Path) -> None:
    fake = FakeHailo(busy_attempts=10)
    waits: list[float] = []
    e = HefEngine(hef, attempts=3, sleep=waits.append, platform_module=fake.module())  # type: ignore[arg-type]
    with pytest.raises(HailoRTException, match="HAILO_OUT_OF_PHYSICAL_DEVICES"):
        e.open()
    assert waits == [1.0, 2.0]  # no wait after the last attempt
    assert "VDevice.release" not in fake.log  # nothing was opened, nothing to release


def test_other_device_errors_are_not_retried(hef: Path) -> None:
    fake = FakeHailo(fail_on="VDevice")
    waits: list[float] = []
    e = HefEngine(hef, sleep=waits.append, platform_module=fake.module())  # type: ignore[arg-type]
    with pytest.raises(HailoRTException, match="fake failure in VDevice"):
        e.open()
    assert waits == []


def test_attempts_must_be_positive(hef: Path) -> None:
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        HefEngine(hef, attempts=0)


def test_infer_sends_one_uint8_frame_as_a_batch(hef: Path) -> None:
    fake = FakeHailo()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    with engine(hef, fake) as e:
        nms = e.infer(frame)
    (sent,) = fake.infer_inputs
    assert list(sent) == ["net/input_layer1"]
    assert sent["net/input_layer1"].shape == (1, 640, 640, 3)
    assert sent["net/input_layer1"].dtype == np.uint8
    best = int(np.argmax(nms.scores))
    assert nms.class_ids[best] == 1
    np.testing.assert_allclose(nms.boxes[best], [0.25, 0.25, 0.75, 0.50])


def test_detect_returns_boxes_on_the_original_image(hef: Path) -> None:
    # 1280 x 720 into 640 x 640: scale 0.5, 140 rows of padding on top. The fake's box covers
    # x 160..320 and y 160..480 in model-input pixels.
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    with engine(hef, FakeHailo()) as e:
        (det,) = e.detect(image, score_threshold=0.5)
    assert det.class_id == 1
    assert det.score == pytest.approx(0.9)
    assert (det.x1, det.y1, det.x2, det.y2) == pytest.approx((320, 40, 640, 680))


def test_infer_on_a_closed_engine_is_refused(hef: Path) -> None:
    e = engine(hef, FakeHailo())
    with pytest.raises(RuntimeError, match="engine is not open"):
        e.infer(np.zeros((640, 640, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "frame",
    [np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((640, 640, 3), dtype=np.float32)],
)
def test_infer_refuses_a_frame_that_does_not_match_the_input(
    hef: Path, frame: npt.NDArray[np.generic]
) -> None:
    fake = FakeHailo()
    with engine(hef, fake) as e, pytest.raises(ValueError, match=r"model input must be uint8"):
        e.infer(frame)  # type: ignore[arg-type]
    assert "infer" not in fake.log
