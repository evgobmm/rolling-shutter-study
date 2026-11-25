"""Run Grid24-RSC with its fitted coarse and 24-region corrections."""

import ctypes as ct
from pathlib import Path

import numpy as np

from ..frames import HEIGHT, WIDTH
from ..runtime import Predictor


def load(assets):
    path = assets / "native" / "grid24.dll"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Build it with python -m rsc.prepare --zig PATH.")
    library = ct.CDLL(str(path.resolve()))
    pointer, integer = ct.c_void_p, ct.c_int
    for name, arguments, result in (
        ("ext_new", (pointer, integer, integer, integer), pointer),
        ("ext_grid", (pointer, pointer), None),
        ("ext_correct", (pointer,) * 6, None),
        ("ext_delete", (pointer,), None),
    ):
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result

    parameters = Path(__file__).with_name("parameters") / "grid24.npz"
    with np.load(parameters) as fit:
        coarse, grid = (np.ascontiguousarray(fit[key], dtype=np.float32) for key in ("coarse", "detail"))
    output = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    state = None

    def close():
        nonlocal state
        if state is not None:
            library.ext_delete(state)
            state = None

    def reset():
        nonlocal state
        close()
        state = library.ext_new(coarse.ctypes.data, 6, 8, 0)
        if not state:
            raise RuntimeError("Could not allocate Grid24-RSC state.")
        library.ext_grid(state, grid.ctypes.data)

    def predict(frames):
        # The previous frame's measurements are retained in the state.
        inputs = (frame.ctypes.data if frame is not None else None for frame in frames)
        library.ext_correct(state, *inputs, output.ctypes.data)
        return output

    reset()
    return Predictor(predict, (0, 1, 2, 3), "CPU", reset=reset, close=close, optional_lookahead=True)
