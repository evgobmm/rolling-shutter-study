"""Fit the six coarse Grid24-RSC shifts using the BS-RSC training scenes."""

import ctypes as ct

import cv2 as cv
import numpy as np

from ..frames import FRAME_COUNT, read, scenes
from . import matched_points
from .tmcd import fit_tmcd


FRAMES = (6, 12, 18, 24, 30, 36, 42)


def coarse_basis(x, y):
    u = np.clip((x - 224) / 288, -.4, 2.4)
    v = np.clip((y - 232) / 312, -.4, 1.4)
    left = (u > 1).astype(int)
    basis = np.zeros((len(x), 6))
    rows = np.arange(len(x))
    for j in range(2):
        for i in range(2):
            basis[rows, left + i + 3*j] = (u - left if i else 1 - u + left) * (v if j else 1 - v)
    return basis


def initial_target(distorted, reference):
    ref, src = (cv.resize(img, (512, 384), interpolation=cv.INTER_AREA) for img in (reference, distorted))
    points, tracks = matched_points(ref, src)
    if len(points) < 35:
        raise ValueError("Too few reference correspondences for the Grid24 training target")
    basis = coarse_basis(2 * points[:, 0], 2 * points[:, 1])
    design = basis * (.45 * (tracks[:, 1, None] / 383.5 - .5) * 4)
    target = tracks - points
    weights = np.ones(len(points))
    penalty = .035 * (np.eye(6) - np.ones((6, 6)) / 6) + .0001 * np.eye(6)
    for _ in range(5):
        fitted = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                 design.T @ (weights[:, None] * target))
        errors = np.linalg.norm(design @ fitted - target, axis=1)
        weights = np.minimum(1., .60 / np.maximum(errors, .001))
    return fitted.astype(np.float32)


def pixel_samples(rng, scale):
    height, width = 768 // scale, 1024 // scale
    rows, cols = np.indices((48, 64))
    px = (cols * width // 64 + rng.integers(width // 64, size=cols.shape)).ravel()
    py = (rows * height // 48 + rng.integers(height // 48, size=rows.shape)).ravel()
    x, y = (px + .5) * scale - .5, (py + .5) * scale - .5
    stripe = (x // 128 + .5) * 128
    return px, py, coarse_basis(stripe, y) * ((y - 383.5) / scale)[:, None]


def photometric_errors(image, truth, sample, velocity, derivatives=False):
    px, py, basis = sample
    denominator = 767 + 3.6 * velocity[:, 1]
    multiplier = 3.6 / denominator
    x = np.clip(px + basis @ (multiplier * velocity[:, 0]), 0, image.shape[1] - 1.00001)
    y = np.clip(py + basis @ (multiplier * velocity[:, 1]), 0, image.shape[0] - 1.00001)
    ix, iy = x.astype(int), y.astype(int)
    rx, ry = (x - ix)[:, None], (y - iy)[:, None]
    a, b = image[iy, ix].astype(float), image[iy, ix+1].astype(float)
    c, d = image[iy+1, ix].astype(float), image[iy+1, ix+1].astype(float)
    upper, lower = a + rx * (b-a), c + rx * (d-c)
    errors = upper + ry * (lower-upper) - truth
    if not derivatives:
        return errors
    dx, dy = (1-ry) * (b-a) + ry * (d-c), lower - upper
    jac = np.empty((len(px), 3, 6, 2))
    jac[..., 0] = dx[:, :, None] * (basis * multiplier)[:, None, :]
    jac[..., 1] = (dy[:, :, None] * (basis * (multiplier * 767 / denominator))[:, None, :]
                   - dx[:, :, None] * (basis * (multiplier**2 * velocity[:, 0]))[:, None, :])
    return errors, jac.reshape(-1, 12)


def refine_target(image, reference, initial, rng):
    velocity = initial.copy()
    for scale in (2, 1):
        size = 1024 // scale, 768 // scale
        source, ref = (cv.resize(frame, size, interpolation=cv.INTER_AREA) if scale == 2 else frame
                       for frame in (image, reference))
        sample = pixel_samples(rng, scale)
        truth = ref[sample[1], sample[0]].astype(float)
        anchor = .7 if scale == 2 else .4

        def loss(candidate):
            errors = photometric_errors(source, truth, sample, candidate)
            return np.mean(errors**2) + anchor * np.mean((candidate-initial)**2)

        for _ in range(5 if scale == 2 else 4):
            errors, jac = photometric_errors(source, truth, sample, velocity, derivatives=True)
            gram = jac.T @ jac / errors.size
            gradient = jac.T @ errors.ravel() / errors.size + anchor * (velocity-initial).ravel() / 12
            damped = gram + np.diag(np.diag(gram) * .12 + .1 + anchor / 12)
            step = np.linalg.solve(damped, gradient).reshape(6, 2)
            fraction = min(1., 4 / max(.001, abs(step).max()))
            current = loss(velocity)
            for factor in (1., .5, .2):
                proposal = velocity - (factor * fraction) * step
                if loss(proposal) < current:
                    velocity = proposal
                    break
            else:
                break
        if scale == 1 and loss(velocity) > loss(initial):
            velocity = initial
    return velocity


def training_motion(directories, assets):
    lib = ct.CDLL(str((assets / "native" / "grid24.dll").resolve()))
    lib.wide_new.argtypes, lib.wide_new.restype = [], ct.c_void_p
    lib.wide_read.argtypes, lib.wide_read.restype = [ct.c_void_p] * 4, None
    lib.wide_delete.argtypes, lib.wide_delete.restype = [ct.c_void_p], None
    samples = []
    for scene in directories:
        state = lib.wide_new()
        if not state:
            raise MemoryError("Could not allocate the Grid24 motion estimator")
        try:
            pairs = np.empty((FRAME_COUNT-1, 7, 2), np.float32)
            current = read(scene, "RS", 0)
            for index, pair in enumerate(pairs):
                following = read(scene, "RS", index+1)
                lib.wide_read(state, current.ctypes.data, following.ctypes.data, pair.ctypes.data)
                current = following
            samples.extend(pairs[np.asarray(FRAMES)[:, None] + np.arange(-2, 2)])
        finally:
            lib.wide_delete(state)
    return np.array(samples, dtype=np.float64)


def fit(data, assets):
    # Preserve the training order used with the fixed sampling seeds.
    directories = sorted(scenes(data, "train"))
    moves = training_motion(directories, assets)
    cases = [(scene, index) for scene in directories for index in FRAMES]
    design = np.column_stack((moves.reshape(len(moves), 56), np.ones(len(moves))))
    prior = np.zeros((12, 57))
    columns = np.arange(56).reshape(4, 7, 2)
    for region in range(6):
        for axis in range(2):
            prior[2*region+axis, columns[:, region+1, axis]] = (0., .5, .5, 0.)

    def ridge(target, weights):
        gram = design.T @ (design * weights[:, None]) + np.eye(57) * weights.sum() * .075
        response = design.T @ ((target - design @ prior.T) * weights[:, None])
        return prior + np.linalg.solve(gram, response).T

    def images(case):
        scene, idx = case
        return read(scene, "RS", idx), read(scene, "GS", idx)

    target = np.array([initial_target(*images(case)) for case in cases], dtype=float).reshape(-1, 12)
    weights = np.ones(len(cases))
    for _ in range(8):
        warm = ridge(target, weights)
        errors = np.sqrt(np.mean((target - design @ warm.T)**2, axis=1))
        weights = np.minimum(1., np.median(errors) / np.maximum(errors, .001))
    start = (design @ warm.astype(np.float32).T).reshape(-1, 6, 2)
    rng = np.random.default_rng(461012)
    target = np.array([refine_target(*images(case), velocity, rng) for case, velocity in zip(cases, start)])
    residual = (moves[:, :, 1:] - target[:, None]).reshape(-1, 24, 2)
    center, covariance = fit_tmcd(residual)
    covariance = .90 * covariance + np.diag(.10 * np.diag(covariance) + .0009)
    errors = residual.reshape(-1, 48) - center
    distances = ((errors @ np.linalg.inv(covariance)) * errors).sum(axis=1)
    coarse = ridge(target.reshape(-1, 12), 1 / np.maximum(1., distances / np.median(distances)))
    detail = np.zeros((2, 25), np.float32)
    for axis in (0, 1):
        detail[axis, [6+axis, 12+axis]] = .5 * .65
    return {"coarse": coarse.astype(np.float32), "detail": detail}
