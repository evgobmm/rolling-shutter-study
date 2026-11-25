"""Read the released BS-RSC image sequences."""

from pathlib import Path

import cv2


WIDTH, HEIGHT = 1024, 768
FRAME_COUNT = 50
COMMON_FRAMES = range(1, FRAME_COUNT - 1)


def scenes(root, split="test"):
    expected = {"test": 15, "val": 16, "train": 50}[split]
    root = Path(root)
    directories = sorted((root / split).glob("Scene*"), key=lambda path: int(path.name[5:]))
    if len(directories) != expected:
        raise ValueError(f"Expected {expected} {split} scenes under {root.resolve()}, found {len(directories)}")
    return directories


def read(scene, kind, frame):
    path = scene / kind / f"{frame:05d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Wrong BS-RSC image dimensions in {path}: {image.shape}")
    return image


def decode_window(scene, first, last, offsets):
    start = max(0, first + min(offsets))
    stop = min(FRAME_COUNT, last + max(offsets) + 1)
    return {index: read(scene, "RS", index) for index in range(start, stop)}
