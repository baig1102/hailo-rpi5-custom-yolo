import importlib
from pathlib import Path

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
