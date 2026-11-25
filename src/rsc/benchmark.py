"""Run the published baselines and the two proposed methods on BS-RSC."""

import argparse
from contextlib import nullcontext
import csv
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns

import cv2 as cv
import numpy as np
from threadpoolctl import threadpool_limits

from .frames import COMMON_FRAMES, FRAME_COUNT, decode_window, read, scenes
from .metrics import score
from .runtime import Predictor


BASELINES = ("vdrsc", "jamnet", "dfrsc-2f", "uni-rife")
OUR_METHODS = ("dis-rsc", "grid24-rsc")
METHODS = (*BASELINES, *OUR_METHODS)


def load_method(name, assets, device):
    if name == "vdrsc":
        from .baselines.vdrsc import rectify_pair
        return Predictor(rectify_pair, (0, 1), "CPU")
    if name in ("jamnet", "uni-rife"):
        from .baselines.pairs import load
        return load(name, assets, device)
    if name == "dfrsc-2f":
        from .baselines.dfrsc import load
        return load(assets, device)
    if name == "dis-rsc":
        from .methods.dis_rsc import load
        return load(device)
    if name == "grid24-rsc":
        from .methods.grid24 import load
        return load(assets)
    raise ValueError(f"Unknown method {name}")


def assess_quality(method, predictor, directories, quick, writer):
    psnr, ssim, paper_psnr, paper_ssim = [], [], [], []
    for scene in directories:
        if predictor.reset:
            predictor.reset()
        first = max(0, -min(predictor.offsets))
        last = FRAME_COUNT - 1 - (1 if predictor.optional_lookahead else max(predictor.offsets))
        if quick:
            first, last = 4, 8
        video = decode_window(scene, first, last, predictor.offsets)
        for index in range(first, last + 1):
            image = predictor.predict(video, index)
            if index == 0 or (quick and index < 6):
                continue
            reference = read(scene, "GS", index)
            if image.shape != reference.shape or image.dtype != np.uint8:
                raise ValueError(f"Wrong prediction format for {scene.name}/{index}")
            p, s = score(image, reference, compact=method in OUR_METHODS)
            paper = () if method in OUR_METHODS else (
                predictor.author_score(reference) if predictor.author_score else (p, s)
            )
            writer.writerow((scene.name, index, p, s, *(paper or ("", ""))))
            if paper:
                paper_psnr.append(paper[0])
                paper_ssim.append(paper[1])
            if index in COMMON_FRAMES:
                psnr.append(p)
                ssim.append(s)
    message = (f"{method} ({predictor.backend}) | common {len(psnr)} frames "
               f"{mean(psnr):.4f} dB / {mean(ssim):.6f} SSIM")
    if paper_psnr:
        aggregate = median if method == "vdrsc" else mean
        message += (f" | paper scoring {len(paper_psnr)} frames "
                    f"{aggregate(paper_psnr):.4f} dB / {aggregate(paper_ssim):.6f} SSIM")
    print(message)


def assess_time(method, predictor, directories, quick, writer):
    values = []
    repeats = 4 if method == "grid24-rsc" and not quick else 1
    for repeat in range(1, repeats + 1):
        for scene_index, scene in enumerate(directories):
            if predictor.reset:
                predictor.reset()
            first = (6, 18, 30)[scene_index % 3]
            last = first + (2 if quick else 7)
            video = decode_window(scene, first - 2, last, predictor.offsets)
            for index in (first - 2, first - 1):
                predictor.predict(video, index)
            for index in range(first, last + 1):
                start = perf_counter_ns()
                predictor.predict(video, index)
                milliseconds = (perf_counter_ns() - start) / 1e6
                writer.writerow((repeat, scene.name, index, milliseconds, predictor.backend))
                values.append(milliseconds)
    print(f"{method} ({predictor.backend}) | {len(values)} timed frames, mean {mean(values):.3f} ms/frame")


def run(args):
    requested = ([args.method] if args.method else
                 {"all": METHODS, "baselines": BASELINES, "ours": OUR_METHODS}[args.suite])
    directories = scenes(args.data)
    if args.quick:
        directories = directories[:1]
    args.output.mkdir(parents=True, exist_ok=True)
    print("Quality and runtime from this run. See results/ for the saved report measurements.", flush=True)
    for method in requested:
        cv.setNumThreads(1 if method == "grid24-rsc" else 4)
        cv.ocl.setUseOpenCL(False)
        predictor = load_method(method, args.assets, args.device)
        paths = [
            (assess_quality, "quality", ("scene", "frame", "psnr_db", "ssim", "paper_psnr_db", "paper_ssim")),
            (assess_time, "timing", ("repeat", "scene", "frame", "milliseconds", "backend")),
        ]
        try:
            with threadpool_limits(limits=2, user_api="blas") if method != "grid24-rsc" else nullcontext():
                for assess, suffix, fields in paths:
                    if args.phase not in (suffix, "both"):
                        continue
                    path = args.output / f"{method}-{suffix}.csv"
                    with path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(fields)
                        assess(method, predictor, directories, args.quick, writer)
        finally:
            if predictor.close:
                predictor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--method", choices=METHODS)
    target.add_argument("--suite", choices=("all", "baselines", "ours"))
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument("--quick", action="store_true", help="Three frames from the first test scene.")
    scale.add_argument("--full", action="store_true", help="All test scenes and the report timing schedule.")
    parser.add_argument("--phase", choices=("quality", "timing", "both"), default="both")
    parser.add_argument("--device", choices=("recorded", "cpu"), default="recorded",
                        help="Use the report's per-method backends, or run everything on CPU.")
    parser.add_argument("--data", type=Path, default=Path("data/BS-RSC"))
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
