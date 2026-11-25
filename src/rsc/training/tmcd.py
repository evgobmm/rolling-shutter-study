"""The NumPy TMCD fit used for the two calibrations. Observations are in C order."""

from math import ceil, prod

import numpy as np
from scipy.stats import chi2


def mode_product(values, matrix, axis):
    return np.moveaxis(np.moveaxis(values, axis, -1) @ matrix.T, -1, axis)


def whiteners(factors):
    return [np.linalg.solve(np.linalg.cholesky(factor), np.eye(len(factor))) for factor in factors]


def log_volume(factors):
    return sum(np.linalg.slogdet(factor)[1] / len(factor) for factor in factors)


def fit_factors(values, initial=None, steps=12):
    center = values.mean(axis=0)
    residual = values - center
    dims = center.shape
    factors = [np.eye(size) for size in dims] if initial is None else [f.copy() for f in initial]
    previous = log_volume(factors)
    for _ in range(steps):
        transforms = whiteners(factors)
        for axis, size in enumerate(dims):
            block = residual
            for other, transform in enumerate(transforms):
                if other != axis:
                    block = mode_product(block, transform, other + 1)
            flat = np.moveaxis(block, axis + 1, -1).reshape(-1, size)
            factor = flat.T @ flat / (len(values) * (center.size // size))
            factors[axis] = (factor + factor.T) * .5
            transforms[axis] = np.linalg.solve(np.linalg.cholesky(factors[axis]), np.eye(size))
        for factor in factors[:-1]:
            scale = np.exp(np.linalg.slogdet(factor)[1] / len(factor))
            factor /= scale
            factors[-1] *= scale
        current = log_volume(factors)
        if abs(previous - current) < 1e-7:
            break
        previous = current
    return center, factors


def distances(values, fit):
    center, factors = fit
    errors = values - center
    for axis, transform in enumerate(whiteners(factors), 1):
        errors = mode_product(errors, transform, axis)
    flat = errors.reshape(len(values), -1)
    return np.einsum("ij,ij->i", flat, flat)


def concentrate(values, fit, count, iterations, fit_steps):
    previous = None
    for _ in range(iterations):
        scores = distances(values, fit)
        selected = np.sort(np.argpartition(scores, count - 1)[:count])
        if np.array_equal(previous, selected):
            break
        volume = log_volume(fit[1])
        fit = fit_factors(values[selected], fit[1], steps=fit_steps)
        if abs(log_volume(fit[1]) - volume) < 1e-7:
            break
        previous = selected
    return fit, selected


def consistency(fraction, dimensions):
    if fraction == 1:
        return 1.
    return fraction / chi2.cdf(chi2.ppf(fraction, dimensions), dimensions + 2)


def fit_tmcd(values):
    values = np.asarray(values, dtype=np.float64)
    dims = values.shape[1:]
    dimensions = prod(dims)
    count = int(len(values) * .75)
    start_size = min(count, max(3, ceil(sum(d*d / dimensions for d in dims)) + 2))
    rng = np.random.default_rng(42)
    candidates = []
    for _ in range(24):
        start = rng.choice(len(values), start_size, replace=False)
        try:
            fit = fit_factors(values[start], steps=3)
            candidates.append(concentrate(values, fit, count, 2, 3)[0])
        except np.linalg.LinAlgError:
            continue
        if len(candidates) == 8:
            break
    if not candidates:
        raise ValueError("No nonsingular TMCD starting subset")
    candidates.sort(key=lambda fit: log_volume(fit[1]))
    refined = [concentrate(values, fit, count, 12, 12) for fit in candidates[:3]]
    raw, support = min(refined, key=lambda item: log_volume(item[0][1]))
    threshold = consistency(.75, dimensions) * chi2.ppf(.975, dimensions)
    keep = np.union1d(support, np.flatnonzero(distances(values, raw) < threshold))
    center, factors = fit_factors(values[keep], raw[1])
    factors[-1] *= consistency(len(keep) / len(values), dimensions)
    covariance = factors[0]
    for factor in factors[1:]:
        covariance = np.kron(covariance, factor)
    return center.ravel(), covariance
