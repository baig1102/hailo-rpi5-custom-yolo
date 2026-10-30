import numpy as np
import numpy.typing as npt
import pytest

from hailo_yolo.preprocess import Letterbox, boxes_to_source, compute_letterbox, picamera2_to_rgb


def to_model_input(boxes: npt.NDArray[np.float64], lb: Letterbox) -> npt.NDArray[np.float64]:
    """Forward mapping, written out by hand so the inverse is tested against it."""
    out = boxes.astype(np.float64)
    out[:, [0, 2]] = out[:, [0, 2]] * lb.scale + lb.pad_left
    out[:, [1, 3]] = out[:, [1, 3]] * lb.scale + lb.pad_top
    return out


def test_landscape_is_padded_top_and_bottom() -> None:
    lb = compute_letterbox(1920, 1080, 640, 640)
    assert lb.scale == pytest.approx(1 / 3)
    assert (lb.new_width, lb.new_height) == (640, 360)
    assert (lb.pad_left, lb.pad_top) == (0, 140)


def test_portrait_is_padded_left_and_right() -> None:
    lb = compute_letterbox(1080, 1920, 640, 640)
    assert (lb.new_width, lb.new_height) == (360, 640)
    assert (lb.pad_left, lb.pad_top) == (140, 0)


def test_exact_fit_has_no_padding() -> None:
    lb = compute_letterbox(1280, 1280, 640, 640)
    assert lb.scale == 0.5
    assert (lb.pad_left, lb.pad_top) == (0, 0)


def test_odd_padding_puts_the_extra_pixel_at_the_bottom() -> None:
    # 640 x 637 into 640 x 640: 3 rows of padding, 1 on top and 2 below.
    lb = compute_letterbox(640, 637, 640, 640)
    assert lb.new_height == 637
    assert lb.pad_top == 1


def test_small_image_is_scaled_up() -> None:
    lb = compute_letterbox(320, 240, 640, 640)
    assert lb.scale == 2.0
    assert (lb.new_width, lb.new_height) == (640, 480)
    assert lb.pad_top == 80


def test_non_square_model_input() -> None:
    lb = compute_letterbox(1920, 1080, 1280, 736)
    assert lb.scale == pytest.approx(2 / 3)
    assert (lb.new_width, lb.new_height) == (1280, 720)
    assert (lb.pad_left, lb.pad_top) == (0, 8)


def test_very_thin_image_keeps_one_pixel() -> None:
    lb = compute_letterbox(4000, 1, 640, 640)
    assert lb.new_height == 1


@pytest.mark.parametrize("sizes", [(0, 480, 640, 640), (640, -1, 640, 640), (640, 480, 0, 640)])
def test_non_positive_size_is_rejected(sizes: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_letterbox(*sizes)


def test_round_trip_returns_the_original_boxes() -> None:
    lb = compute_letterbox(1920, 1080, 640, 640)
    boxes = np.array([[100.0, 200.0, 400.0, 600.0], [0.0, 0.0, 1920.0, 1080.0]])
    np.testing.assert_allclose(boxes_to_source(to_model_input(boxes, lb), lb), boxes)


def test_box_reaching_into_the_padding_is_clipped_to_the_image() -> None:
    lb = compute_letterbox(1920, 1080, 640, 640)
    # y1 = 100 lies in the top padding (0..140), y2 = 520 in the bottom padding (500..640).
    out = boxes_to_source([[10.0, 100.0, 50.0, 520.0]], lb)
    np.testing.assert_allclose(out, [[30.0, 0.0, 150.0, 1080.0]])


def test_empty_boxes_give_an_empty_result() -> None:
    lb = compute_letterbox(1920, 1080, 640, 640)
    assert boxes_to_source(np.zeros((0, 4)), lb).shape == (0, 4)


def test_boxes_with_the_wrong_shape_are_rejected() -> None:
    lb = compute_letterbox(1920, 1080, 640, 640)
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        boxes_to_source([1.0, 2.0, 3.0, 4.0], lb)


def pixel(fmt_channels: str) -> npt.NDArray[np.uint8]:
    """A 1 x 1 frame with R=10, G=20, B=30 (and A=255) in the given channel order."""
    values = {"R": 10, "G": 20, "B": 30, "A": 255}
    return np.array([[[values[c] for c in fmt_channels]]], dtype=np.uint8)


@pytest.mark.parametrize(
    ("fmt", "layout"),
    [("RGB888", "BGR"), ("BGR888", "RGB"), ("XRGB8888", "BGRA"), ("XBGR8888", "RGBA")],
)
def test_picamera2_formats_come_out_as_rgb(fmt: str, layout: str) -> None:
    rgb = picamera2_to_rgb(pixel(layout), fmt)
    assert rgb.tolist() == [[[10, 20, 30]]]
    assert rgb.flags.c_contiguous


def test_unknown_picamera2_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Picamera2 format 'YUV420'"):
        picamera2_to_rgb(pixel("RGB"), "YUV420")


def test_frame_with_the_wrong_channel_count_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"XRGB8888 frames have shape \(H, W, 4\)"):
        picamera2_to_rgb(pixel("RGB"), "XRGB8888")
