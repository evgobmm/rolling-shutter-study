"""Paper-based reconstruction of V-DRSC on CPU."""

from functools import lru_cache

import cv2 as cv
import numpy as np


READOUT_RATIO = 0.45


@lru_cache(maxsize=2)
def image_grid(height, width):
    return np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))


def sparse_flow(source, following):
    height, width = source.shape[:2]
    step = 8
    while (width // step) * (height // step) > np.iinfo(np.int16).max:
        step *= 2
    x, y = image_grid(height, width)
    points = np.column_stack((x[::step, ::step].ravel(), y[::step, ::step].ravel()))
    first = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    second = cv.cvtColor(following, cv.COLOR_BGR2GRAY)
    next_points, forward_status, _ = cv.calcOpticalFlowPyrLK(first, second, points, None)
    valid = forward_status.ravel().astype(bool)
    points, next_points = points[valid], next_points[valid]
    if not len(points):
        return points, next_points, np.empty(0, dtype=np.float32)
    returned, backward_status, _ = cv.calcOpticalFlowPyrLK(second, first, next_points, None)
    errors = np.linalg.norm(returned - points, axis=1)
    errors[backward_status.ravel() == 0] = np.inf
    return points, next_points, errors


def dense_flow(source, following, points, next_points):
    if len(points) < 4:
        return np.zeros((*source.shape[:2], 2), dtype=np.float32)
    interpolator = cv.ximgproc.createEdgeAwareInterpolator()
    interpolator.setK(min(128, len(points)))
    interpolator.setSigma(0.05)
    interpolator.setUsePostProcessing(True)
    interpolator.setFGSLambda(500)
    interpolator.setFGSSigma(1.5)
    return interpolator.interpolate(source, points, following, next_points)


def estimate_checked_flow(source, following, threshold=1.0):
    points, next_points, errors = sparse_flow(source, following)
    checked = errors <= threshold
    return dense_flow(source, following, points[checked], next_points[checked])


def rectify(source, flow, *, target_row=0.5, ratio=READOUT_RATIO, iterations=4):
    height, width = source.shape[:2]
    x, y = image_grid(height, width)
    fx, fy = flow.transpose(2, 0, 1)
    # Equation (12) maps source pixels to GS. Invert it for OpenCV's remap.
    fraction = ratio * (y - target_row * (height - 1)) / (height + ratio * fy)
    displacement = flow * fraction[..., None]
    sample_x, sample_y = x + fraction * fx, y + fraction * fy
    for _ in range(iterations - 1):
        offset = cv.remap(displacement, sample_x, sample_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        sample_x, sample_y = x + offset[..., 0], y + offset[..., 1]
    return cv.remap(source, sample_x, sample_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)


def rectify_pair(images):
    source, following = images
    return rectify(source, estimate_checked_flow(source, following))
