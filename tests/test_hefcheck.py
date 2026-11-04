import subprocess
from pathlib import Path

import pytest

from hailo_yolo.hefcheck import HefInfo, check_hef, main, parse_hef_output
from hailo_yolo.report import Finding

FIXTURES = Path(__file__).parent / "fixtures" / "parse_hef"


def load(name: str) -> HefInfo:
    return parse_hef_output((FIXTURES / name).read_text())


def test_parses_architecture_and_network_group() -> None:
    info = load("yolov8_hailo8l_640.txt")
    assert info.architecture == "HAILO8L"
    assert info.network_groups == ("custom_yolov8s",)


def test_parses_the_input_shape() -> None:
    (inp,) = load("yolov8_hailo8l_640.txt").inputs
    assert inp.name == "custom_yolov8s/input_layer1"
    assert (inp.format_type, inp.format_order) == ("UINT8", "NHWC")
    assert inp.shape == (640, 640, 3)


def test_parses_the_nms_output() -> None:
    (out,) = load("yolov8_hailo8l_640.txt").outputs
    assert out.format_type == "FLOAT32"
    assert out.format_order == "HAILO NMS BY CLASS"
    assert out.shape is None
    assert (out.nms_classes, out.nms_max_bboxes_per_class) == (3, 100)


def test_parses_the_nms_op() -> None:
    nms = load("yolov8_hailo8l_640.txt").nms
    assert nms is not None
    assert nms.op_type == "YOLOV8"
    assert nms.score_threshold == pytest.approx(0.25)
    assert nms.iou_threshold == pytest.approx(0.45)
    assert nms.classes == 3
    assert (nms.image_height, nms.image_width) == (640, 640)


def test_multi_context_header_still_parses() -> None:
    info = load("yolov8_input1280_nms640.txt")
    assert info.network_groups == ("custom_yolov8s",)
    assert info.inputs[0].shape == (1280, 1280, 3)


def test_hef_without_nms() -> None:
    info = load("yolov8_no_nms.txt")
    assert info.nms is None
    assert len(info.outputs) == 6
    assert info.outputs[0].shape == (80, 80, 64)


def test_windows_line_endings_are_accepted() -> None:
    text = (FIXTURES / "yolov8_hailo8l_640.txt").read_text().replace("\n", "\r\n")
    info = parse_hef_output(text)
    assert info.inputs[0].shape == (640, 640, 3)
    assert info.nms is not None
    assert info.nms.image_width == 640


def test_text_that_is_not_parse_hef_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="is this parse-hef output"):
        parse_hef_output("[HailoRT] [error] CHECK failed - Failed opening file")


def levels(findings: list[Finding]) -> list[str]:
    return [f.level for f in findings]


def problems(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level != "info"]


def test_correct_hef_on_the_right_device_has_no_problems() -> None:
    findings = check_hef(load("yolov8_hailo8l_640.txt"), device_arch="HAILO8L")
    assert problems(findings) == []
    assert any("matches the device" in f.message for f in findings)
    assert any("both 640x640" in f.message for f in findings)


def test_hailo8_hef_on_a_hailo8l_device_is_an_error() -> None:
    (arch,) = [
        f for f in check_hef(load("yolov8_hailo8_640.txt"), "HAILO8L") if "device" in f.message
    ]
    assert arch.level == "error"
    assert "won't load on a HAILO8L device" in arch.message


def test_hailo8l_hef_on_a_hailo8_device_is_a_performance_warning() -> None:
    (arch,) = problems(check_hef(load("yolov8_hailo8l_640.txt"), "HAILO8"))
    assert arch.level == "warning"
    assert "lower performance" in arch.message


def test_device_arch_is_case_insensitive() -> None:
    assert problems(check_hef(load("yolov8_hailo8l_640.txt"), "hailo8l")) == []


def test_without_a_device_the_board_name_trap_is_explained() -> None:
    findings = check_hef(load("yolov8_hailo8_640.txt"))
    assert "error" not in levels(findings)
    assert any("not with 'Board Name'" in f.message for f in findings)


def test_unknown_device_is_not_checked() -> None:
    findings = check_hef(load("yolov8_hailo8l_640.txt"), "HAILO15H")
    assert any("not checked here" in f.message for f in findings)


def test_input_size_different_from_nms_size_is_a_warning() -> None:
    (warning,) = problems(check_hef(load("yolov8_input1280_nms640.txt"), "HAILO8L"))
    assert warning.level == "warning"
    assert "1280x1280" in warning.message
    assert "configured for 640x640" in warning.message


def test_hef_without_nms_is_a_warning() -> None:
    (warning,) = problems(check_hef(load("yolov8_no_nms.txt"), "HAILO8L"))
    assert "No built-in NMS" in warning.message


def test_nms_thresholds_are_reported() -> None:
    findings = check_hef(load("yolov8_hailo8l_640.txt"))
    assert any("score threshold 0.25, IoU threshold 0.45" in f.message for f in findings)


def test_cli_from_file_reports_and_exits_0_without_problems(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["--from-file", str(FIXTURES / "yolov8_hailo8l_640.txt"), "--device-arch", "HAILO8L"]
    )
    assert code == 0
    assert "INFO     HEF compiled for HAILO8L matches the device." in capsys.readouterr().out


def test_cli_exits_1_on_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["--from-file", str(FIXTURES / "yolov8_hailo8_640.txt"), "--device-arch", "HAILO8L"]
    )
    assert code == 1
    assert "ERROR    HEF compiled for HAILO8" in capsys.readouterr().out


def test_cli_exits_1_on_a_warning() -> None:
    assert main(["--from-file", str(FIXTURES / "yolov8_input1280_nms640.txt")]) == 1


def test_cli_missing_hef_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.hef")]) == 2
    assert "HEF not found" in capsys.readouterr().err


def test_cli_runs_hailortcli_with_the_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        text = (FIXTURES / "yolov8_hailo8l_640.txt").read_text()
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    assert main(["model.hef", "--device-arch", "HAILO8L"]) == 0
    assert calls == [["hailortcli", "parse-hef", str(hef.resolve())]]


def test_cli_without_hailortcli_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", "hailortcli")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main([str(hef)]) == 2
    assert "is hailo-all installed?" in capsys.readouterr().err


def test_cli_reports_a_failing_hailortcli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HAILO_INVALID_HEF(26)")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main([str(hef)]) == 2
    assert "HAILO_INVALID_HEF" in capsys.readouterr().err
