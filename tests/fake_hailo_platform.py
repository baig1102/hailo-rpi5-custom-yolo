"""A stand-in for the parts of ``hailo_platform`` (HailoRT 4.20) that the engine uses.

It mimics the call signatures in hailo-ai/hailort v4.20.0,
hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py:
``HEF(str)`` (line 323), ``VDevice()`` with ``configure`` and ``release`` (lines 3442-3505),
``ConfiguredNetwork.activate`` (line 556), ``InputVStreamParams.make`` /
``OutputVStreamParams.make`` (lines 3587, 3652) and ``InferVStreams`` (line 852).
Every call is appended to ``log`` so tests can check the order. Nothing here talks to hardware.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np


class HailoRTException(Exception):
    pass


class HailoStreamInterface:
    PCIe = "PCIe"


class FormatType:
    UINT8 = "UINT8"
    FLOAT32 = "FLOAT32"


@dataclass
class FakeHailo:
    """One fake ``hailo_platform`` module, with knobs for the failures the engine must handle."""

    input_shape: tuple[int, int, int] = (640, 640, 3)
    classes: int = 2
    max_boxes: int = 3
    busy_attempts: int = 0  # VDevice() fails this many times with OUT_OF_PHYSICAL_DEVICES
    fail_on: str | None = None  # a log entry name that raises instead of succeeding
    log: list[str] = field(default_factory=list)
    infer_inputs: list[dict[str, Any]] = field(default_factory=list)

    def _call(self, name: str) -> None:
        self.log.append(name)
        if self.fail_on == name:
            raise HailoRTException(f"libhailort failed with error: 8 (fake failure in {name})")

    def module(self) -> SimpleNamespace:
        fake = self

        class HEF:
            def __init__(self, hef_source: object) -> None:
                if not isinstance(hef_source, str):  # like the real one: str path or bytes
                    raise ValueError("HEF can only be created from a file path (str) or a buffer")
                fake._call("HEF")
                self.path = hef_source

            def get_input_vstream_infos(self) -> list[SimpleNamespace]:
                return [SimpleNamespace(name="net/input_layer1", shape=fake.input_shape)]

            def get_output_vstream_infos(self) -> list[SimpleNamespace]:
                return [SimpleNamespace(name="net/yolov8_nms_postprocess", shape=())]

        class ConfiguredNetwork:
            def create_params(self) -> str:
                return "network_group_params"

            def activate(self, params: object) -> "Activation":
                fake._call("activate")
                return Activation()

        class Activation:
            def __enter__(self) -> "Activation":
                fake._call("activation.enter")
                return self

            def __exit__(self, *args: object) -> None:
                fake.log.append("activation.exit")

        class VDevice:
            def __init__(self) -> None:
                if fake.busy_attempts > 0:
                    fake.busy_attempts -= 1
                    fake.log.append("VDevice busy")
                    raise HailoRTException(
                        "libhailort failed with error: 74 (HAILO_OUT_OF_PHYSICAL_DEVICES)"
                    )
                fake._call("VDevice")

            def configure(self, hef: object, params: object) -> list[ConfiguredNetwork]:
                fake._call("configure")
                return [ConfiguredNetwork()]

            def release(self) -> None:
                fake.log.append("VDevice.release")

        class ConfigureParams:
            @staticmethod
            def create_from_hef(hef: object, interface: str) -> dict[str, str]:
                fake.log.append(f"ConfigureParams({interface})")
                return {"net": interface}

        class InputVStreamParams:
            @staticmethod
            def make(network: object, format_type: str) -> str:
                fake.log.append(f"InputVStreamParams({format_type})")
                return "input_params"

        class OutputVStreamParams:
            @staticmethod
            def make(network: object, format_type: str) -> str:
                fake.log.append(f"OutputVStreamParams({format_type})")
                return "output_params"

        class InferVStreams:
            def __init__(
                self, network: object, inputs: object, outputs: object, tf_nms_format: bool
            ) -> None:
                fake.log.append(f"InferVStreams(tf_nms_format={tf_nms_format})")

            def __enter__(self) -> "InferVStreams":
                fake._call("pipeline.enter")
                return self

            def __exit__(self, *args: object) -> None:
                fake.log.append("pipeline.exit")

            def infer(self, input_data: dict[str, Any]) -> dict[str, Any]:
                fake._call("infer")
                fake.infer_inputs.append(input_data)
                (batch,) = input_data.values()
                out = np.zeros((len(batch), fake.classes, 5, fake.max_boxes), dtype=np.float32)
                # One detection of class 1: y_min, x_min, y_max, x_max, score.
                out[:, 1, :, 0] = [0.25, 0.25, 0.75, 0.50, 0.9]
                return {"net/yolov8_nms_postprocess": out}

        return SimpleNamespace(
            HEF=HEF,
            VDevice=VDevice,
            ConfigureParams=ConfigureParams,
            HailoStreamInterface=HailoStreamInterface,
            FormatType=FormatType,
            InputVStreamParams=InputVStreamParams,
            OutputVStreamParams=OutputVStreamParams,
            InferVStreams=InferVStreams,
            HailoRTException=HailoRTException,
        )
