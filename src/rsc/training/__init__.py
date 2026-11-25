"""Training-time correspondences shared by the two correction models."""

import cv2 as cv
import numpy as np


def matched_points(reference, distorted):
    ref = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
    src = cv.cvtColor(distorted, cv.COLOR_BGR2GRAY)
    points = cv.goodFeaturesToTrack(ref, 340, .006, 11)
    if points is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    tracked, forward, error = cv.calcOpticalFlowPyrLK(ref, src, points, None, winSize=(19, 19))
    returned, backward, _ = cv.calcOpticalFlowPyrLK(src, ref, tracked, None, winSize=(19, 19))
    roundtrip = np.linalg.norm((points - returned)[:, 0], axis=1)
    keep = (forward.ravel() == 1) & (backward.ravel() == 1) & (roundtrip < .8) & (error.ravel() < 23)
    return points[keep, 0], tracked[keep, 0]
