"""Common model interface and ONNX Runtime session settings."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Predictor:
    apply: Callable
    offsets: tuple[int, ...]
    backend: str
    author_score: Callable | None = None
    reset: Callable | None = None
    close: Callable | None = None
    optional_lookahead: bool = False

    def predict(self, video, index):
        return self.apply(tuple(video.get(index + offset) for offset in self.offsets))


def onnx_session(path, device):
    import onnxruntime as ort

    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. See REPRODUCE.md for preparing the model exports.")
    providers = ["CPUExecutionProvider"]
    if device == "recorded":
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("The recorded backend requires onnxruntime-directml on Windows. Use --device cpu for a CPU run.")
        providers.insert(0, "DmlExecutionProvider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def rgb_tensor(image):
    rgb = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    return rgb / np.float32(255)


def bgr_image(prediction):
    bgr = prediction[0, ::-1].transpose(1, 2, 0)
    return np.uint8(np.rint(bgr.clip(0, 1) * np.float32(255)))
