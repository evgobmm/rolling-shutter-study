"""DIS-RSC motion estimation and correction with the fitted ridge model."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2 as cv
import numpy as np

from ..runtime import Predictor


RATIO = .45
OFFSETS = np.asarray(((0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)), dtype=np.float32)
PRESETS = {"fast": cv.DISOPTICAL_FLOW_PRESET_FAST}


@lru_cache(maxsize=8)
def grid(height, width):
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return np.stack((x, y), axis=-1)


def warp(image, coordinates):
    return cv.remap(image, coordinates, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)


def resize(image, size, interpolation=cv.INTER_LINEAR):
    return cv.resize(image, size, interpolation=interpolation)


@lru_cache(maxsize=4)
def hann(size):
    return cv.createHanningWindow(size, cv.CV_32F)


@dataclass
class Motion:
    gray: tuple
    flows: tuple
    grid: np.ndarray
    duration: float
    opencl: bool
    reverse: tuple = ()

    def base_map(self, scale=1.0, quadratic=1.0):
        past, future = self.flows
        height = self.grid.shape[0]
        # The corresponding source and destination rows have different readout times.
        dt_past = -1 + RATIO * past[..., 1:2] / (height - 1)
        dt_future = 1 + RATIO * future[..., 1:2] / (height - 1)
        speed_past = past / dt_past
        speed_future = future / dt_future
        acceleration = 2 * (speed_future - speed_past) / (dt_future - dt_past)
        velocity = speed_future - .5 * acceleration * dt_future
        time = RATIO * scale * (self.grid[..., 1:2] / (height - 1) - .5)
        inverse_displacement = time * velocity - .5 * quadratic * time**2 * acceleration
        first = self.grid + inverse_displacement
        return self.grid + warp(inverse_displacement, first)


class Tracker:
    def __init__(self, width, preset, phase=False, *, opencl=False, bidirectional=False, reuse_forward=False):
        self.size = (width, width * 3 // 4)
        self.estimator = cv.DISOpticalFlow_create(PRESETS[preset])
        self.phase = phase
        self.cache = {}
        self.opencl = opencl
        self.bidirectional = bidirectional
        self.reuse_forward = reuse_forward
        self.pairs = {}

    def estimate(self, source, neighbour, source_gpu, neighbour_gpu):
        initial = None
        shift = None
        if self.phase:
            candidate, response = cv.phaseCorrelate(source.astype(np.float32), neighbour.astype(np.float32), hann(self.size))
            if response > .17 and np.linalg.norm(candidate) < .35 * self.size[0]:
                shift = np.asarray(candidate, dtype=np.float32)
        if self.opencl:
            if shift is not None:
                matrix = np.asarray(((1., 0., shift[0]), (0., 1., shift[1])), dtype=np.float32)
                neighbour_gpu = cv.warpAffine(neighbour_gpu, matrix, self.size,
                                              flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                                              borderMode=cv.BORDER_REPLICATE)
            result = self.estimator.calc(source_gpu, neighbour_gpu, cv.UMat()).get()
            return result if shift is None else result + shift
        if shift is not None:
            initial = np.empty((*neighbour.shape, 2), dtype=np.float32)
            initial[:] = shift
        return self.estimator.calc(source, neighbour, initial)

    def __call__(self, images):
        start = perf_counter()
        gray = []
        cache = {}
        for image in images:
            key = id(image)
            if key not in self.cache:
                channel = cv.cvtColor(resize(image, self.size, cv.INTER_AREA), cv.COLOR_BGR2GRAY)
                item = image, channel, cv.UMat(channel) if self.opencl else None
            else:
                item = self.cache[key]
            cache[key] = item
            gray.append(item[1])
        self.cache = cache
        pairs = {}
        order = ((1, 0), (1, 2))
        if self.bidirectional:
            order += ((0, 1), (2, 1))
        keys = tuple(id(image) for image in images)
        for left, right in order:
            key = keys[left], keys[right]
            estimate = self.pairs.get(key)
            # In a video, the previous call already computed frame t-1 to t.
            if estimate is None and self.reuse_forward and (left, right) == (1, 0):
                forward = self.pairs.get((key[1], key[0]))
                if forward is not None:
                    points = grid(*gray[1].shape)
                    estimate = -forward
                    for _ in range(2):
                        estimate = -warp(forward, points + estimate)
            if estimate is None:
                estimate = self.estimate(gray[left], gray[right], cache[keys[left]][2], cache[keys[right]][2])
            pairs[key] = estimate
        self.pairs = pairs
        estimates = tuple(pairs[keys[left], keys[right]] for left, right in order)
        return Motion(tuple(gray), estimates[:2], grid(*gray[1].shape), perf_counter() - start, self.opencl, estimates[2:])


def native_coordinates(coordinates, shape):
    height, width = shape
    small_h, small_w = coordinates.shape[:2]
    shift = resize(coordinates - grid(small_h, small_w), (width, height))
    shift *= (width / small_w, height / small_h)
    return grid(height, width) + shift


@lru_cache(maxsize=2)
def opencl_grid(height, width):
    return cv.UMat(grid(height, width))


def native_warp(image, coordinates, opencl):
    height, width = image.shape[:2]
    if not opencl:
        return warp(image, native_coordinates(coordinates, (height, width)))
    small_h, small_w = coordinates.shape[:2]
    shift = cv.UMat(coordinates - grid(small_h, small_w))
    shift = resize(shift, (width, height))
    shift = cv.multiply(shift, (width / small_w, height / small_h, 0., 0.))
    mapping = cv.add(opencl_grid(height, width), shift)
    return warp(cv.UMat(image), mapping).get()


def feature_tensor(motion, points):
    small_h, small_w = motion.grid.shape[:2]
    t = RATIO * (points[..., 1:2] / (small_h - 1) - .5)
    sides = []
    for sign, flow in zip((-1, 1), motion.flows):
        neighbourhood = np.stack([warp(flow, points + offset) for offset in OFFSETS], axis=-2)
        interval = sign + RATIO * neighbourhood[..., 1:2] / (small_h - 1)
        sides.append(t[..., None] * neighbourhood / interval)
    return np.stack(sides, axis=-3)


def predictors(motion, points, base, observations, appearance=False):
    h, w = motion.grid.shape[:2]
    xy = points / np.asarray((w - 1, h - 1), dtype=np.float32) - .5
    offset = warp(base - motion.grid, points)
    flat = observations.reshape(*points.shape[:2], -1)
    diff = observations[..., 1, 0, :] - observations[..., 0, 0, :]
    magnitude = np.linalg.norm(diff, axis=-1, keepdims=True)
    features = [flat, xy, xy**2, offset, diff, magnitude]
    if appearance:
        gray = motion.gray[1].astype(np.float32) / 255
        tone = warp(gray, points)
        local = warp(cv.Laplacian(gray, cv.CV_32F, ksize=3), points)
        signal = [tone, local]
        for i, source in enumerate((0, 2)):
            coords = points + warp(motion.flows[i], points)
            neighbour = warp(motion.gray[source], coords).astype(np.float32) / 255
            signal.append(np.abs(tone - neighbour))
        features.extend(item[..., None] for item in signal)
    return np.concatenate(features, axis=-1)


def design(values, center, scale):
    normalized = (values - center) / scale
    return np.concatenate((normalized, np.ones((*values.shape[:-1], 1), dtype=np.float32)), axis=-1)


def residual_features(motion, base, center, scale, appearance=False):
    points = np.ascontiguousarray(motion.grid[::4, ::4])
    return design(predictors(motion, points, base, feature_tensor(motion, points), appearance), center, scale)


def predict_residual(model, features, size):
    vectors = features.reshape(-1, features.shape[-1])
    if isinstance(model, np.ndarray):
        predictions = vectors @ model
    else:
        predictions = model.predict(vectors)
    low = np.clip(predictions.reshape(*features.shape[:2], 2), -6, 6).astype(np.float32)
    return resize(low, size)


def load(device):
    opencl = device == "recorded"
    if opencl:
        cv.ocl.setUseOpenCL(True)
        if not cv.ocl.useOpenCL():
            raise RuntimeError("DIS-RSC could not enable OpenCL. Use --device cpu or check the OpenCL driver.")
    parameters = Path(__file__).with_name("parameters") / "dis_rsc.npz"
    with np.load(parameters) as fit:
        center, scale, coefficients = (fit[name] for name in ("center", "scale", "coefficients"))
        gain = float(fit["gain"])
    tracker = Tracker(256, "fast", True, opencl=opencl, reuse_forward=True)

    def predict(images):
        motion = tracker(images)
        coordinates = motion.base_map(1., 0.)
        features = residual_features(motion, coordinates, center, scale)
        coordinates += gain * predict_residual(coefficients, features, tracker.size)
        return native_warp(images[1], coordinates, opencl)

    def reset():
        tracker.cache.clear()
        tracker.pairs.clear()

    return Predictor(predict, (-1, 0, 1), "CPU + OpenCL" if opencl else "CPU", reset=reset)
