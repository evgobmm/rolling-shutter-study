"""Reproduce the visual comparison of BS-RSC Scene57 frame 00023."""

import argparse
from contextlib import nullcontext
import csv
from pathlib import Path

import cv2 as cv
from threadpoolctl import threadpool_limits

from .benchmark import load_method
from .frames import decode_window, read
from .metrics import score


def corrected_frame(name, assets, scene, frame, device):
    cv.setNumThreads(1 if name == "grid24-rsc" else 4)
    cv.ocl.setUseOpenCL(False)
    predictor = load_method(name, assets, device)
    try:
        first = max(0, -min(predictor.offsets)) if predictor.reset else frame
        video = decode_window(scene, first, frame, predictor.offsets)
        if predictor.reset:
            predictor.reset()
        limits = nullcontext() if name == "grid24-rsc" else threadpool_limits(limits=2, user_api="blas")
        with limits:
            for index in range(first, frame + 1):
                image = predictor.predict(video, index)
        return image
    finally:
        if predictor.close:
            predictor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ada-rsc", type=Path, required=True)
    parser.add_argument("--dfrsc-3f", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/BS-RSC"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--device", choices=("recorded", "cpu"), default="recorded")
    parser.add_argument("--output", type=Path, default=Path("runs/scene57"))
    args = parser.parse_args()

    scene, frame = args.data / "test" / "Scene57", 23
    reference = read(scene, "GS", frame)
    images = {
        "input-rs": read(scene, "RS", frame),
        "ada-rsc": cv.imread(str(args.ada_rsc), cv.IMREAD_COLOR),
        "dfrsc-3f": cv.imread(str(args.dfrsc_3f), cv.IMREAD_COLOR),
    }
    for name in ("ada-rsc", "dfrsc-3f"):
        if images[name] is None or images[name].shape != reference.shape:
            raise ValueError(f"Expected a full-resolution Scene57 frame 00023 for {name}.")
    for name in ("uni-rife", "grid24-rsc", "dis-rsc"):
        images[name] = corrected_frame(name, args.assets, scene, frame, args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    images["ground-truth"] = reference
    with (args.output / "quality.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "psnr_db", "ssim"))
        for name, image in images.items():
            if not cv.imwrite(str(args.output / f"{name}.png"), image):
                raise OSError(f"Could not write {name}.png")
            if name != "ground-truth":
                writer.writerow((name, *score(image, reference)))


if __name__ == "__main__":
    main()
