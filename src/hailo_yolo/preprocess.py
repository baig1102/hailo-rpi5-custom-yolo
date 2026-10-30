"""Prepare camera frames for the model: colour order and letterbox geometry.

YOLO models are trained on letterboxed RGB images: scaled to fit, aspect ratio kept, the rest
padded. Inference has to do the same, and boxes from the model then have to be mapped back
through the padding and the scale to land on the original image.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Letterbox:
    """How a source image was fitted into the model input."""

    src_width: int
    src_height: int
    scale: float
    new_width: int
    new_height: int
    pad_left: int
    pad_top: int


def compute_letterbox(
    src_width: int, src_height: int, dst_width: int, dst_height: int
) -> Letterbox:
    """Compute scale and padding to fit a source image into a dst_width x dst_height input.

    Small images are scaled up, and the padding is split as evenly as possible with an odd
    extra pixel on the right or bottom. Both match the defaults of Ultralytics' ``LetterBox``
    (``ultralytics/data/augment.py``: ``scaleup=True``, ``left = round(dw / 2 - 0.1)``), so the
    geometry matches what the model saw in training.
    """
    for name, value in (
        ("src_width", src_width),
        ("src_height", src_height),
        ("dst_width", dst_width),
        ("dst_height", dst_height),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    scale = min(dst_width / src_width, dst_height / src_height)
    # At least 1 pixel, so a very thin image doesn't vanish (Ultralytics does the same).
    new_width = max(round(src_width * scale), 1)
    new_height = max(round(src_height * scale), 1)
    return Letterbox(
        src_width=src_width,
        src_height=src_height,
        scale=scale,
        new_width=new_width,
        new_height=new_height,
        pad_left=(dst_width - new_width) // 2,
        pad_top=(dst_height - new_height) // 2,
    )


def boxes_to_source(boxes: npt.ArrayLike, lb: Letterbox) -> npt.NDArray[np.float64]:
    """Map (N, 4) boxes ``[x1, y1, x2, y2]`` from model-input pixels to source-image pixels.

    Boxes are clipped to the source image, so a box that reaches into the padding ends at
    the image edge.
    """
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {arr.shape}")

    out = np.empty_like(arr)
    out[:, [0, 2]] = (arr[:, [0, 2]] - lb.pad_left) / lb.scale
    out[:, [1, 3]] = (arr[:, [1, 3]] - lb.pad_top) / lb.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, lb.src_width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, lb.src_height)
    return out


# Channel order of the numpy array Picamera2 returns for each format. It is the reverse of the
# format name: "BGR888" gives R, G, B. Source: Picamera2 manual, Appendix A "Pixel and image
# formats", Table 1.
_PICAMERA2_CHANNELS = {
    "RGB888": "BGR",
    "BGR888": "RGB",
    "XRGB8888": "BGRA",
    "XBGR8888": "RGBA",
}


def picamera2_to_rgb(frame: npt.NDArray[np.uint8], fmt: str) -> npt.NDArray[np.uint8]:
    """Return a Picamera2 frame as a contiguous (H, W, 3) RGB array, the order YOLO expects.

    ``fmt`` is the format string the camera was configured with, e.g. ``"BGR888"``. The alpha
    channel of the 4-channel formats is dropped.
    """
    channels = _PICAMERA2_CHANNELS.get(fmt)
    if channels is None:
        raise ValueError(
            f"unsupported Picamera2 format {fmt!r}, use one of {list(_PICAMERA2_CHANNELS)}"
        )
    if frame.ndim != 3 or frame.shape[2] != len(channels):
        raise ValueError(f"{fmt} frames have shape (H, W, {len(channels)}), got {frame.shape}")

    rgb = frame[:, :, :3] if channels.startswith("RGB") else frame[:, :, 2::-1]
    return np.ascontiguousarray(rgb)


def letterbox_image(
    image: npt.NDArray[np.uint8], dst_width: int, dst_height: int, pad_value: int = 114
) -> tuple[npt.NDArray[np.uint8], Letterbox]:
    """Resize and pad an (H, W, C) image into the model input; return it with its geometry.

    Linear interpolation and grey padding (114) are the defaults of Ultralytics' ``LetterBox``,
    so the input looks like the training images. Keep the returned ``Letterbox`` to map the
    model's boxes back with ``boxes_to_source``.
    """
    # Imported here so the rest of the package works without OpenCV installed.
    import cv2

    if image.ndim != 3:
        raise ValueError(f"image must have shape (H, W, C), got {image.shape}")
    src_height, src_width, channels = image.shape
    lb = compute_letterbox(src_width, src_height, dst_width, dst_height)

    if (lb.new_width, lb.new_height) == (src_width, src_height):
        resized = image
    else:
        # cv2's stubs type the result loosely; it is uint8 like the input, so this doesn't copy.
        resized = np.asarray(
            cv2.resize(image, (lb.new_width, lb.new_height), interpolation=cv2.INTER_LINEAR),
            dtype=np.uint8,
        )

    out = np.full((dst_height, dst_width, channels), pad_value, dtype=np.uint8)
    out[lb.pad_top : lb.pad_top + lb.new_height, lb.pad_left : lb.pad_left + lb.new_width] = resized
    return out, lb
