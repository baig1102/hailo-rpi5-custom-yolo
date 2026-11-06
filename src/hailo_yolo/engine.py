"""Run a HEF with built-in NMS on the legacy ``InferVStreams`` API.

This is the path verified on my Pi 5 + Hailo-8L with HailoRT 4.20: ``HEF`` -> ``VDevice`` ->
``ConfigureParams.create_from_hef(..., PCIe)`` -> ``configure`` -> ``activate`` ->
``InferVStreams(..., tf_nms_format=True)``. Call signatures are from hailo-ai/hailort v4.20.0,
hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py.

``hailo_platform`` is imported only when an engine opens, so the package (and its tests) work
without Hailo software. Create and use an engine in one process: ``VDevice.configure`` refuses
to run in another process than the one that created the device (pyhailort.py line 3506).
"""

import importlib
import logging
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from hailo_yolo.postprocess import Detection, NmsDetections, from_tf_format, to_detections
from hailo_yolo.preprocess import letterbox_image

log = logging.getLogger(__name__)


class HefEngine:
    """Owns the device, the configured network and the inference pipeline for one HEF.

    Use it as a context manager. Shutdown always runs in reverse order of opening: stop the
    pipeline, deactivate the network, release the device. On my setup, releasing the device
    while inference still ran ended in ``HAILO_STREAM_ABORT``.

    Not thread-safe: let one worker thread own the engine (open, every ``infer``, close). A
    frame that is sent while another thread reconfigures the camera or closes the engine is
    how ``HAILO_TIMEOUT`` and ``HAILO_STREAM_ABORT`` happened on my setup.
    """

    def __init__(
        self,
        hef_path: str | Path,
        *,
        attempts: int = 5,
        first_delay_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        platform_module: ModuleType | None = None,
    ) -> None:
        """Try to get the device up to ``attempts`` times, doubling the wait each time.

        The defaults wait 1 + 2 + 4 + 8 = 15 s in total, long enough for a crashed process to
        let go of the chip. ``sleep`` and ``platform_module`` (instead of ``hailo_platform``)
        are for tests.
        """
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        self.hef_path = Path(hef_path).expanduser().resolve()
        self.attempts = attempts
        self.first_delay_s = first_delay_s
        self._sleep = sleep
        self._platform = platform_module
        self._stack: ExitStack | None = None
        self._pipeline: Any = None
        self.input_name = ""
        self.output_name = ""
        self.input_shape: tuple[int, int, int] = (0, 0, 0)  # (H, W, C)

    def open(self) -> "HefEngine":
        if self._stack is not None:
            raise RuntimeError("engine is already open")
        # A relative path breaks as soon as the app starts from another directory
        # (e.g. from systemd), so the path is absolute and checked before HailoRT sees it.
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"HEF not found: {self.hef_path}")
        hp = self._platform or importlib.import_module("hailo_platform")

        with ExitStack() as stack:
            hef = hp.HEF(str(self.hef_path))  # HEF() takes a str path, not a Path
            vdevice = self._open_vdevice(hp)
            stack.callback(vdevice.release)

            params = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
            networks = vdevice.configure(hef, params)
            if len(networks) != 1:
                raise ValueError(f"expected 1 network group in the HEF, found {len(networks)}")
            network = networks[0]

            inputs = hef.get_input_vstream_infos()
            outputs = hef.get_output_vstream_infos()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(
                    f"expected 1 input and 1 NMS output, found {len(inputs)} and {len(outputs)}"
                )
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
            height, width, channels = inputs[0].shape
            self.input_shape = (int(height), int(width), int(channels))

            input_params = hp.InputVStreamParams.make(network, format_type=hp.FormatType.UINT8)
            # FLOAT32 output: HailoRT dequantises the NMS boxes and scores for us.
            output_params = hp.OutputVStreamParams.make(network, format_type=hp.FormatType.FLOAT32)
            stack.enter_context(network.activate(network.create_params()))
            self._pipeline = stack.enter_context(
                hp.InferVStreams(network, input_params, output_params, tf_nms_format=True)
            )
            # Everything opened: keep it open until close().
            self._stack = stack.pop_all()
        return self

    def infer(self, model_input: npt.NDArray[np.uint8]) -> NmsDetections:
        """Run one frame that is already letterboxed RGB of exactly ``input_shape``."""
        if self._pipeline is None:
            raise RuntimeError("engine is not open: use it as a context manager or call open()")
        if model_input.shape != self.input_shape or model_input.dtype != np.uint8:
            raise ValueError(
                f"model input must be uint8 {self.input_shape}, "
                f"got {model_input.dtype} {model_input.shape}"
            )
        batch = np.ascontiguousarray(model_input)[np.newaxis]  # HailoRT wants (N, H, W, C)
        outputs = self._pipeline.infer({self.input_name: batch})
        return from_tf_format(outputs[self.output_name][0])

    def detect(
        self, rgb_image: npt.NDArray[np.uint8], *, score_threshold: float
    ) -> list[Detection]:
        """Letterbox an RGB image of any size, run it, and return boxes on that image."""
        height, width, _ = self.input_shape
        model_input, lb = letterbox_image(rgb_image, width, height)
        return to_detections(self.infer(model_input), lb, score_threshold=score_threshold)

    def _open_vdevice(self, hp: Any) -> Any:
        """``VDevice()``, retried while another process holds the chip.

        HailoRT 4.20 has no own exception class for this status: it arrives as a plain
        ``HailoRTException`` whose message names it (pyhailort.py line 161).
        """
        delay = self.first_delay_s
        for attempt in range(1, self.attempts + 1):
            try:
                return hp.VDevice()
            except hp.HailoRTException as e:
                if "HAILO_OUT_OF_PHYSICAL_DEVICES" not in str(e) or attempt == self.attempts:
                    raise
                log.warning(
                    "Hailo device busy (attempt %d of %d), retrying in %.1f s",
                    attempt,
                    self.attempts,
                    delay,
                )
                self._sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # the loop either returns or raises

    def close(self) -> None:
        """Stop the pipeline, deactivate, release the device. Safe to call more than once."""
        stack, self._stack, self._pipeline = self._stack, None, None
        if stack is not None:
            stack.close()

    def __enter__(self) -> "HefEngine":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
