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
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any


class HefEngine:
    """Owns the device, the configured network and the inference pipeline for one HEF.

    Use it as a context manager. Shutdown always runs in reverse order of opening: stop the
    pipeline, deactivate the network, release the device. Releasing the device while
    inference still runs ends in ``HAILO_STREAM_ABORT``.
    """

    def __init__(self, hef_path: str | Path, *, platform_module: ModuleType | None = None) -> None:
        """``platform_module`` replaces ``hailo_platform``; only tests pass it."""
        self.hef_path = Path(hef_path).expanduser().resolve()
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
            vdevice = hp.VDevice()
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
