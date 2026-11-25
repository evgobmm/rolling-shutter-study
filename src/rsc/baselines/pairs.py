"""JAMNet and Uni-RIFE inference and exports from the released BS-RSC weights."""

from functools import lru_cache
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np

from ..frames import HEIGHT, WIDTH
from ..metrics import score_float_rgb
from ..runtime import Predictor, bgr_image, onnx_session


def export_path(method, assets):
    return assets / "onnx" / f"{method}_bsrsc_native.onnx"


def load(method, assets, device):
    session = onnx_session(export_path(method, assets), device)
    raw = None

    def predict(images):
        nonlocal raw
        rgb = np.concatenate([frame[..., ::-1] for frame in images], axis=2)
        inputs = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
        raw = session.run(["rgb"], {"input": inputs})[0]
        return bgr_image(raw)

    def author_score(reference):
        return score_float_rgb(raw, reference)

    return Predictor(predict, (-1, 0), "DirectML" if device == "recorded" else "CPU", author_score)


def source_model(method, assets):
    import torch

    @lru_cache(maxsize=4)
    def cpu_grid(height, width):
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing="ij")
        return torch.stack((x, y))

    if method == "jamnet":
        from torchvision.ops import DeformConv2d

        class ModulatedDeformConv(DeformConv2d):
            def __init__(self, *args, deformable_groups=1, **kwargs):
                # torchvision obtains the deformable group count from the offsets.
                if deformable_groups < 1:
                    raise ValueError("Invalid deformable group count")
                super().__init__(*args, **kwargs)

        root = assets / "upstream" / "JAMNet"
        sys.path[:0] = [str(root / "deep_unroll_net"), str(root / "package_core")]
        detectron = ModuleType("detectron2")
        layers = ModuleType("detectron2.layers")
        layers.ModulatedDeformConv = layers.DeformConv = ModulatedDeformConv
        with patch.dict(sys.modules, {"detectron2": detectron, "detectron2.layers": layers}):
            import network_JAM as source
        source.generate_2D_grid = cpu_grid
        model = source.JAMNet()
        checkpoint = root / "deep_unroll_weights" / "model_weights" / "bsrsc" / "pre_net_flow.pth"
    else:
        root = assets / "upstream" / "UniVR" / "UniVR_RIFE" / "deep_unroll_net"
        sys.path[:0] = [str(root), str(root / "RIFE")]
        import network_UniVR as source
        source.generate_2D_grid = cpu_grid
        model = source.UniVR()
        checkpoint = assets / "checkpoints" / "unirife_bsrsc_pre_net_flow.pth"

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)
    return model.eval()


def export(method, assets):
    import torch

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    net = source_model(method, assets)

    class Restore(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, inputs):
            if method == "jamnet":
                return self.network(inputs)[0]
            return self.network(inputs, timestep=0.775, gamma=0.45, scale_list=[4, 2, 1])[0][-1]

    path = export_path(method, assets)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(Restore(net).eval(), torch.zeros(1, 6, HEIGHT, WIDTH), path,
                          opset_version=16, input_names=["input"], output_names=["rgb"], dynamo=False)
    print(f"Wrote {path}")
