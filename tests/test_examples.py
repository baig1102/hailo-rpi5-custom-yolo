import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
from fake_hailo_platform import FakeHailo

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
