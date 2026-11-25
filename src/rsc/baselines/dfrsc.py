"""DFRSC-2F with its four warp fields and reusable per-frame features."""

from functools import lru_cache
import importlib.util
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from ..frames import HEIGHT, WIDTH
from ..runtime import Predictor, bgr_image, onnx_session, rgb_tensor


LEVELS = ("level4", "level3", "level2", "level1", "level0")


def paths(assets):
    return tuple(assets / "onnx" / f"dfrsc_2f_{stage}.onnx" for stage in ("encoder", "decoder"))


def load(assets, device):
    encoder, decoder = (onnx_session(path, device) for path in paths(assets))
    cached = {}

    def predict(frames):
        nonlocal cached
        current = {}
        for frame in frames:
            key = id(frame)
            if key not in cached:
                features = encoder.run(list(LEVELS), {"image": rgb_tensor(frame)})
                current[key] = frame, features
            else:
                current[key] = cached[key]
        cached = current
        features = (cached[id(frame)][1] for frame in frames)
        feeds = {name: np.stack(pair, axis=1) for name, pair in zip(LEVELS, zip(*features))}
        return bgr_image(decoder.run(["rgb"], feeds)[0])

    return Predictor(predict, (-1, 0), "DirectML" if device == "recorded" else "CPU")


def install_compact_warp(model, source):
    import torch
    from torch.nn import functional as F

    @lru_cache(maxsize=8)
    def grid(height, width):
        y, x = torch.meshgrid(torch.linspace(-1., 1., height), torch.linspace(-1., 1., width), indexing="ij")
        return torch.stack((x, y), dim=-1)

    def warp(image, flow):
        height, width = map(int, flow.shape[2:])
        x = grid(height, width)[..., 0] + flow[:, 0] / ((width - 1.) / 2.)
        y = grid(height, width)[..., 1] + flow[:, 1] / ((height - 1.) / 2.)
        return F.grid_sample(image, torch.stack((x, y), dim=-1).to(image.dtype),
                             padding_mode="border", align_corners=True)

    source.backward_warp = warp

    def forward(decoder, x, rs_imgs, flow_t, feats_t):
        batch, frames, channels, height, width = x.shape
        warped = torch.stack([warp(x[:, i], flow_t[:, i]) for i in range(frames)], 1).flatten(1, 2)
        base = torch.cat((feats_t.flatten(1, 2), warped, flow_t.flatten(1, 2)), dim=1)
        flows = decoder.pred_multi_flow(base).reshape(batch, frames, decoder.num_flows, 2, height, width)
        inputs = x.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        fused = None
        for i in range(decoder.num_flows):
            field = flows[:, :, i].reshape(batch * frames, 2, height, width)
            weight = decoder.fusion_conv.weight[:, i * channels:(i + 1) * channels]
            part = F.conv2d(warp(inputs, field), weight, bias=decoder.fusion_conv.bias if i == 0 else None)
            fused = part if fused is None else fused + part
        features = torch.cat((feats_t.flatten(1, 2),
                              fused.reshape(batch, frames * channels, height, width),
                              flows.flatten(1, 3)), dim=1)
        return decoder.pred_img(decoder.convblock(features))

    model.multi_flow_decoder.forward = MethodType(forward, model.multi_flow_decoder)


def export(assets):
    import torch

    torch.set_num_threads(3)
    source_path = assets / "upstream" / "DFRSC" / "model" / "dfrsc.py"
    spec = importlib.util.spec_from_file_location("upstream_dfrsc", source_path)
    source = importlib.util.module_from_spec(spec)
    registry = ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = SimpleNamespace(register=lambda: lambda cls: cls)
    with patch.dict(sys.modules, {"basicsr.utils.registry": registry}):
        spec.loader.exec_module(source)
    model = source.DFRSC(inner_channels=[16, 28, 40, 64, 96], num_frames=2, num_flows=4)
    checkpoint = torch.load(assets / "checkpoints" / "dfrsc_2f_bsrsc.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    install_compact_warp(model, source)
    model.eval().to(memory_format=torch.channels_last)

    class Encoder(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.encoder = network.encoder

        def forward(self, image):
            return tuple(self.encoder(image))

    class Decoder(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
            rows = torch.arange(HEIGHT).float() / (HEIGHT - 1)
            center = (HEIGHT - 1) / 2
            rows = rows - rows[int(center)] - (center % 1) / (HEIGHT - 1)
            rows = rows.reshape(1, 1, HEIGHT, 1).expand(2, 1, HEIGHT, WIDTH)
            self.register_buffer("rows", rows[None].contiguous(), persistent=False)

        def forward(self, level4, level3, level2, level1, level0):
            flow, features = self.network.pred_init_flow(level4, t_embedding=self.rows, split=True)
            stages = (self.network.decoder4, self.network.decoder3, self.network.decoder2, self.network.decoder1)
            for stage, level in zip(stages, (level4, level3, level2, level1)):
                flow, features = stage(level, flow, features, split=True)
            return self.network.multi_flow_decoder(level0, None, flow, features)

    encoder, decoder = paths(assets)
    encoder.parent.mkdir(parents=True, exist_ok=True)
    features = tuple(torch.zeros(1, 2, channels, HEIGHT // divisor, WIDTH // divisor)
                     for channels, divisor in zip((96, 64, 40, 28, 16), (16, 8, 4, 2, 1)))
    with torch.no_grad():
        torch.onnx.export(Encoder(model).eval(), torch.zeros(1, 3, HEIGHT, WIDTH), encoder,
                          opset_version=16, input_names=["image"], output_names=list(LEVELS), dynamo=False)
        torch.onnx.export(Decoder(model).eval(), features, decoder,
                          opset_version=16, input_names=list(LEVELS), output_names=["rgb"], dynamo=False)
    print(f"Wrote {encoder} and {decoder}")
