"""PSNR and Gaussian-window SSIM in RGB."""

from functools import lru_cache

import cv2 as cv
import numpy as np


@lru_cache(maxsize=1)
def author_kernel():
    kernel = np.exp(-((np.arange(11, dtype=np.float32) - 5.5) / 1.5) ** 2 / 2)
    return kernel / kernel.sum()


def similarity(a, b, gaussian, c1, c2):
    mean_a, mean_b = gaussian(a), gaussian(b)
    aa, bb, ab = mean_a * mean_a, mean_b * mean_b, mean_a * mean_b
    var_a, var_b = gaussian(a * a) - aa, gaussian(b * b) - bb
    cov = gaussian(a * b) - ab
    score = ((2 * ab + c1) * (2 * cov + c2)) / ((aa + bb + c1) * (var_a + var_b + c2))
    return float(score.mean(dtype=np.float64))


def score(image, reference, *, compact=False):
    dtype = np.float32 if compact else np.float64
    window = cv.getGaussianKernel(11, 1.5)

    def gaussian(frame):
        return cv.sepFilter2D(frame, -1, window, window)[5:-5, 5:-5]

    psnr = cv.PSNR(image, reference) if not np.array_equal(image, reference) else float("inf")
    ssim = similarity(image.astype(dtype), reference.astype(dtype), gaussian,
                      dtype((.01 * 255) ** 2), dtype((.03 * 255) ** 2))
    return psnr, ssim


def score_float_rgb(prediction, reference):
    """JAMNet and Uni-RIFE score float output with zero-padded windows."""
    rgb = np.ascontiguousarray(prediction[0].transpose(1, 2, 0))
    target = np.ascontiguousarray(reference[..., ::-1], dtype=np.float32) / 255
    window = author_kernel()

    def gaussian(frame):
        return cv.sepFilter2D(frame, -1, window, window, borderType=cv.BORDER_CONSTANT)

    mse = np.mean((rgb - target) ** 2, dtype=np.float64)
    psnr = 100.0 if mse == 0 else float(-10 * np.log10(mse))
    return psnr, similarity(rgb, target, gaussian, .01**2, .03**2)
