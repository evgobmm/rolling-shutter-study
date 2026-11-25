"""Refit the reported ridge parameters on the official BS-RSC training scenes."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from threadpoolctl import threadpool_limits

from ..frames import read, scenes
from ..methods.dis_rsc import Tracker, design, feature_tensor, predictors, warp
from . import grid24, matched_points
from .tmcd import fit_tmcd


def fit_dis(data):
    tracker = Tracker(256, "fast", phase=True)
    rng = np.random.default_rng(3872)
    features, targets, tensors = [], [], []
    for scene in sorted(scenes(data, "train")):
        for index in (10, 20, 30, 40):
            frames = tuple(read(scene, "RS", j) for j in (index-1, index, index+1))
            reference = read(scene, "GS", index)
            motion = tracker(frames)
            base = motion.base_map(1., 0.)
            ref, src = matched_points(reference, frames[1])
            if not len(ref):
                continue
            rows = rng.choice(len(ref), min(180, len(ref)), replace=False)
            points = (ref[rows] * .25)[None]
            expected = (src[rows] - ref[rows]) * .25
            observed = feature_tensor(motion, points)
            shift = warp(base - motion.grid, points)[0]
            features.append(predictors(motion, points, base, observed)[0])
            targets.append(expected - shift)
            scale = .7 + .16 * np.linalg.norm(shift, axis=-1)
            tensors.append((observed[0] - expected[:, None, None, :]) / scale[:, None, None, None])
    raw, target, residual = (np.concatenate(parts) for parts in (features, targets, tensors))
    center = np.median(raw, axis=0).astype(np.float32)
    scale = np.maximum(raw.std(axis=0), .06).astype(np.float32)
    matrix = design(raw, center, scale)
    rng = np.random.default_rng(908)
    sample = rng.choice(len(residual), min(6500, len(residual)), replace=False)
    location, covariance = fit_tmcd(residual[sample])
    covariance = .85 * covariance + np.diag(.15 * np.diag(covariance) + .003)
    errors = residual.reshape(len(raw), -1) - location.astype(np.float32)
    precision = np.linalg.inv(covariance).astype(np.float32)
    distances = np.maximum((errors * (errors @ precision)).sum(axis=1), 0)
    cutoff = np.quantile(distances, .75)
    weights = np.minimum(1., np.sqrt(cutoff / np.maximum(distances, 1e-5))).astype(np.float32)
    gram = matrix.T @ (matrix * weights[:, None]) / len(raw)
    response = matrix.T @ (target * weights[:, None]) / len(raw)
    coefficients = np.linalg.solve(gram + .025 * np.eye(gram.shape[0]), response).astype(np.float32)
    return {"center": center, "scale": scale, "coefficients": coefficients, "gain": 1.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("dis-rsc", "grid24-rsc"))
    parser.add_argument("--data", type=Path, default=Path("data/BS-RSC"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("runs/refit"))
    args = parser.parse_args()
    cv.ocl.setUseOpenCL(False)
    cv.setNumThreads(4)
    dense = args.method == "dis-rsc"
    with threadpool_limits(limits=2 if dense else 1, user_api="blas"):
        fitted = fit_dis(args.data) if dense else grid24.fit(args.data, args.assets)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("dis_rsc.npz" if dense else "grid24.npz")
    np.savez(path, **fitted)
    print(f"Saved training parameters to {path}")


if __name__ == "__main__":
    main()
