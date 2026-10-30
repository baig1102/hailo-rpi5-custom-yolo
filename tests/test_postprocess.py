import numpy as np
import numpy.typing as npt
import pytest

from hailo_yolo.postprocess import from_hailo_format, from_tf_format

# Two classes. Class 0 has two boxes, class 1 has one. Rows: y_min, x_min, y_max, x_max, score.
CLASS0 = [[0.10, 0.20, 0.30, 0.40, 0.90], [0.50, 0.55, 0.60, 0.65, 0.40]]
CLASS1 = [[0.05, 0.15, 0.25, 0.35, 0.70]]


def test_hailo_format_flattens_classes_in_order() -> None:
    nms = from_hailo_format([np.array(CLASS0), np.array(CLASS1)])
    assert nms.class_ids.tolist() == [0, 0, 1]
    np.testing.assert_allclose(nms.boxes, [r[:4] for r in CLASS0 + CLASS1])
    np.testing.assert_allclose(nms.scores, [0.90, 0.40, 0.70])


def test_hailo_format_class_without_detections() -> None:
    # HailoRT gives an empty class an array of shape (0, 5).
    nms = from_hailo_format([np.empty((0, 5)), np.array(CLASS1)])
    assert nms.class_ids.tolist() == [1]
    np.testing.assert_allclose(nms.scores, [0.70])


def test_hailo_format_no_detections_at_all() -> None:
    nms = from_hailo_format([np.empty((0, 5)), np.empty((0, 5))])
    assert nms.boxes.shape == (0, 4)
    assert nms.scores.shape == (0,)
    assert nms.class_ids.shape == (0,)


def test_hailo_format_rejects_a_wrongly_shaped_class() -> None:
    with pytest.raises(ValueError, match=r"class 1: expected shape \(detections, 5\)"):
        from_hailo_format([np.array(CLASS0), np.zeros((2, 4))])


def build_tf_frame(max_boxes: int, empty_value: float) -> npt.NDArray[np.float64]:
    """The same detections in the TF layout (classes, 5, max_boxes), padded like HailoRT does."""
    frame = np.full((2, max_boxes, 5), empty_value)
    frame[0, :2] = CLASS0
    frame[1, :1] = CLASS1
    return frame.transpose(0, 2, 1)


def test_tf_format_box_values_run_along_the_middle_axis() -> None:
    frame = build_tf_frame(max_boxes=3, empty_value=0.0)
    assert frame.shape == (2, 5, 3)
    nms = from_tf_format(frame)
    assert nms.class_ids.tolist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(nms.boxes[0], CLASS0[0][:4])
    np.testing.assert_allclose(nms.boxes[3], CLASS1[0][:4])
    np.testing.assert_allclose(nms.scores, [0.90, 0.40, 0.0, 0.70, 0.0, 0.0])


def test_tf_format_keeps_padding_slots_even_if_not_exactly_zero() -> None:
    # A dequantised "zero" box can be slightly off zero; it must still come out as a slot,
    # to be dropped later by the score threshold.
    nms = from_tf_format(build_tf_frame(max_boxes=3, empty_value=0.003))
    np.testing.assert_allclose(nms.scores, [0.90, 0.40, 0.003, 0.70, 0.003, 0.003])


def test_both_layouts_agree_on_the_real_detections() -> None:
    hailo = from_hailo_format([np.array(CLASS0), np.array(CLASS1)])
    tf = from_tf_format(build_tf_frame(max_boxes=4, empty_value=0.0))
    real = tf.scores > 0
    np.testing.assert_array_equal(tf.class_ids[real], hailo.class_ids)
    np.testing.assert_allclose(tf.boxes[real], hailo.boxes)
    np.testing.assert_allclose(tf.scores[real], hailo.scores)


@pytest.mark.parametrize("shape", [(2, 3), (2, 4, 3), (1, 2, 5, 3)])
def test_tf_format_rejects_other_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"shape \(classes, 5, max_boxes\)"):
        from_tf_format(np.zeros(shape))
