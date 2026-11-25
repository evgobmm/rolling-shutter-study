"""Prepare the released model checkpoints and build the Grid24-RSC library."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile


REVISIONS = {
    "jamnet": ("JAMNet", "GitCVfb/JAMNet", "7a4756c155ad559c32384c9ef213f1d0b7e9da62"),
    "dfrsc-2f": ("DFRSC", "ljzycmd/DFRSC", "adde28a1e7e9b287d451dba7402fff2005d94315"),
    "uni-rife": ("UniVR", "GitCVfb/UniVR", "cd520a86dbd65cfcd3024a1434a9e2daf2303cfe"),
}
DFRSC_WEIGHTS = "https://github.com/ljzycmd/DFRSC/releases/download/ckpts/dfrsc_2f_bsrsc.pth"
DFRSC_SHA256 = "236919653fa45544b9e91ba94bc1c9b6ed311f0b48bbfc3a1655d83955c50007"
UNIRIFE_SHA256 = "f634310f8e122c4a7073071bfcd7f2244bd000dea96682f472cd8cf95bd3aef9"
UNIRIFE_MEMBER = "UniVR_RIFE/deep_unroll_weights/pre_bsrsc_ft/pre_net_flow.pth"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path):
    request = Request(url, headers={"User-Agent": "rolling-shutter-study"})
    with urlopen(request, timeout=60) as response, path.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def fetch_source(method, assets):
    name, project, commit = REVISIONS[method]
    destination = assets / "upstream" / name
    if destination.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as stage:
        stage = Path(stage)
        archive = stage / "source.zip"
        print(f"Downloading {project} at {commit}", flush=True)
        download(f"https://codeload.github.com/{project}/zip/{commit}", archive)
        with ZipFile(archive) as zipped:
            zipped.extractall(stage)
        (stage / f"{name}-{commit}").rename(destination)


def fetch_checkpoint(method, assets, univr_archive):
    if method == "jamnet":
        return
    folder = assets / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    filename, expected = (("dfrsc_2f_bsrsc.pth", DFRSC_SHA256) if method == "dfrsc-2f"
                          else ("unirife_bsrsc_pre_net_flow.pth", UNIRIFE_SHA256))
    target = folder / filename
    if target.is_file() and sha256(target) == expected:
        return
    with tempfile.TemporaryDirectory(dir=folder) as stage:
        candidate = Path(stage) / filename
        if method == "dfrsc-2f":
            download(DFRSC_WEIGHTS, candidate)
        else:
            if univr_archive is None:
                raise FileNotFoundError("Pass --univr-archive with the Google Drive checkpoint ZIP linked in REPRODUCE.md.")
            with ZipFile(univr_archive) as zipped:
                names = [name for name in zipped.namelist() if name.endswith(UNIRIFE_MEMBER)]
                if len(names) != 1:
                    raise ValueError("Could not find the Uni-RIFE BS-RSC checkpoint in the archive.")
                with zipped.open(names[0]) as source, candidate.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if sha256(candidate) != expected:
            raise ValueError(f"Checkpoint checksum mismatch for {filename}")
        candidate.replace(target)


def build_grid24(zig, assets):
    source = Path(__file__).with_name("methods") / "native" / "extended_shutter.c"
    path = assets / "native" / "grid24.dll"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as stage:
        candidate = Path(stage) / path.name
        subprocess.run(
            [zig, "cc", "-O3", "-mavx2", "-mfma", "-shared", "-o", str(candidate), str(source)],
            check=True,
        )
        candidate.replace(path)
    print(f"Built {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--baseline", choices=tuple(REVISIONS))
    command.add_argument("--zig", metavar="PATH", help="Path to Zig 0.15.2, or zig if it is on PATH.")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--univr-archive", type=Path)
    args = parser.parse_args()
    assets = args.assets.resolve()
    if args.zig:
        build_grid24(args.zig, assets)
        return
    fetch_checkpoint(args.baseline, assets, args.univr_archive)
    fetch_source(args.baseline, assets)
    if args.baseline == "dfrsc-2f":
        from .baselines.dfrsc import export

        export(assets)
    else:
        from .baselines.pairs import export

        export(args.baseline, assets)


if __name__ == "__main__":
    main()
