# Reproduce the report

Open the executed notebooks directly on GitHub.

- The [baseline notebook](notebooks/01_published_baselines.ipynb) reproduces the four measured rows on page 4 from the [per-frame quality scores](results/quality.csv) and [timings](results/timings.csv).
- The [proposed-methods notebook](notebooks/02_our_methods.ipynb) calculates the common-frame scores, speedups and one-minute clip estimates on page 6 and in the one-pager. The two published-only rows come from the cited papers.
- The [scene 57 CSV](results/scene57-frame00023.csv) gives the scores for the visual comparison on page 7.

## Run on Windows

Run the commands from the repository root. Use 64-bit Python 3.12, an AVX2/FMA-capable CPU, and [Zig 0.15.2 for Windows](https://ziglang.org/download/0.15.2/zig-x86_64-windows-0.15.2.zip) on PATH. The recorded configuration uses a plugged-in Ryzen 9 5900HX laptop with integrated Radeon graphics.

Download the authors’ [BS-RSC archive](https://drive.google.com/file/d/1h7UP1kci8zbg3TQp37J-imrvzlh2X6zn/view?usp=share_link). Extract its test split under `data/BS-RSC/test`, with files such as `Scene51/RS/00000.png` and matching `GS` frames. Download the [UniVR checkpoint archive](https://drive.google.com/file/d/1HJBkGR9P4rNbOqJ-NtPYbsnGxSUkLupp/view?usp=sharing). Set its path in the third preparation command below.

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -e '.[windows,notebooks,export]'
& $python -m rsc.prepare --baseline jamnet
& $python -m rsc.prepare --baseline dfrsc-2f
& $python -m rsc.prepare --baseline uni-rife --univr-archive 'C:\path\to\UniVR-checkpoints.zip'
& $python -m rsc.prepare --zig zig
$env:OPENCV_OPENCL_BUILD_EXTRA_OPTIONS = '-cl-std=CL2.0'
& $python -m rsc.benchmark --suite all --full --device recorded --output runs/benchmark
```

Use `--quick` for a three-frame check. The [benchmark](src/rsc/benchmark.py) writes separate quality and timing CSVs for each method under `runs/benchmark/`. The notebooks open with the archived results. Their last cell can run the benchmark again. The OpenCL option selects the DIS-RSC configuration measured on the test Radeon.

Common quality uses frames 1–48 in each of the 15 test scenes. The paper checks use each method’s scoring convention. Timing covers RAM input to RAM output after decoding, including transfers. Each scene has two warm-up frames and eight timed frames. Grid24-RSC repeats this schedule four times.

## Refit the two models

The benchmark uses the [fitted parameters](src/rsc/methods/parameters). To regenerate them, extract the 50 BS-RSC training scenes under `data/BS-RSC/train` and run the [training code](src/rsc/training) with the settings selected on validation. It writes new parameters under `runs/refit/`.

```powershell
& $python -m pip install -e '.[training]'
& $python -m rsc.training --method dis-rsc
& $python -m rsc.training --method grid24-rsc
```

## Visual comparison

The example uses OpenCV's default compiler options. On the test Radeon, DIS optical flow falls back to the CPU and the warp runs on the GPU.

Extract scene 57, frame 00023 from each of the two published archives and pass the image paths below. The command saves the input, reference and five corrected images, then scores the input and predictions against the global-shutter reference. [AdaRSC predictions](https://github.com/ljzycmd/BSRSC/releases/download/v1.0.0/Results_on_BSRSC.zip) · [DFRSC-3F predictions](https://github.com/ljzycmd/DFRSC/releases/download/results/DFRSC_3F_BSRSC.zip)

```powershell
Remove-Item Env:OPENCV_OPENCL_BUILD_EXTRA_OPTIONS -ErrorAction SilentlyContinue
& $python -m rsc.frame_example --ada-rsc 'C:\path\to\AdaRSC\Scene57\00023.png' --dfrsc-3f 'C:\path\to\DFRSC\Scene57\00023.png' --output runs/scene57
```
