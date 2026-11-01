from pathlib import Path

import pytest

from hailo_yolo.hefcheck import HefInfo, parse_hef_output

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
