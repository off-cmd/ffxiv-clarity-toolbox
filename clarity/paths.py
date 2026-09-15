"""Where things live, derived from the project layout (P:\\meta-index\\STRUCTURE.md, TREE-FULL.md).

    P:\\projects\\ffxiv-texture-upscale\\
    ├── scripts-local\\clarity-upscale\\      this package (tracked)
    │   └── models\\registry*.json           model-slot config (tracked)
    ├── build-output\\manifest.sqlite        the working database (ignored, hash-guarded)
    ├── analysis-specimens\\models\\          model weights, as obtained
    ├── analysis-specimens\\reslogger\\       CurrentPathList-<date>.gz from rl2.perchbird.dev
    ├── vendor-tools\\texconv\\texconv.exe    a *release* DirectXTex texconv
    └── hash-manifests\\                     texture-fingerprints.tsv (exported by `clarity fingerprint --export`)


Every one of these is a default, not a rule: the CLARITY_* variable in the right-hand column wins.
The layout is walked from this file's location, so the package works wherever the project is
mounted (P: on Windows, /sessions/.../mnt/projects on Linux) without a config file.
"""

import glob
import os
from typing import Optional

PACKAGE = os.path.dirname(os.path.abspath(__file__))
TOOL_ROOT = os.path.normpath(os.path.join(PACKAGE, ".."))  # the repository root
PROJECT = TOOL_ROOT
PROJECTS = os.path.normpath(os.path.join(PROJECT, ".."))  # P:\workspaces


def _env(name, default):
    v = os.environ.get(name)
    return v if v else default


# Where texconv's per-texture scratch files go.
#
# This is NOT cosmetic. `_texconv` hands the encoder an *uncompressed* RGBA DDS of the already
# upscaled image, and Python's tempfile defaults to %TEMP%, which on Windows is on C:. A 4x pass
# over a 2048-square source writes 8192x8192x4 = 256 MiB of input plus ~85 MiB of BC7 output, per
# texture, to the system drive -- so the upscale step churns C: even when the project, the models,
# the database and the output folder are all on P:. That is why the low-space warning arrives during
# upscaling and never during packing: packing writes to --out, and cmd_run's disk advisory measures
# --out, so neither of them is looking at the drive actually filling up.
#
# The default keeps it inside the project. Point CLARITY_SCRATCH at a RAM disk or a scratch SSD if
# the write volume matters: the files are written once, read once, and deleted, so this is pure
# churn -- at roughly 340 MiB a texture it dwarfs the ~404 GB of real output.
SCRATCH = _env("CLARITY_SCRATCH", os.path.join(PROJECT, "temp-scratch", "texconv"))

DB = _env("CLARITY_DB", os.path.join(PROJECT, "build-output", "manifest.sqlite"))
MODELS = _env("CLARITY_MODELS", os.path.join(PROJECT, "analysis-specimens", "models"))
REGISTRY = _env("CLARITY_REGISTRY", os.path.join(TOOL_ROOT, "models", "registry.json"))
TEXCONV = _env(
    "CLARITY_TEXCONV", os.path.join(PROJECT, "vendor-tools", "texconv", "texconv.exe")
)

FINGERPRINTS = _env(
    "CLARITY_FINGERPRINTS",
    os.path.join(PROJECT, "hash-manifests", "texture-fingerprints.tsv"),
)
RESLOGGER_DIR = os.path.join(PROJECT, "analysis-specimens", "reslogger")


def newest_pathlist():
    """The most recent ResLogger list in analysis-specimens\\reslogger, or None."""
    env = os.environ.get("CLARITY_PATHLIST")
    if env:
        return env
    cands = sorted(glob.glob(os.path.join(RESLOGGER_DIR, "CurrentPathList*.gz")))
    return cands[-1] if cands else None


def describe():
    rows = [
        ("db", DB),
        ("models", MODELS),
        ("registry", REGISTRY),
        ("texconv", TEXCONV),
        ("scratch", SCRATCH),
        
        ("fingerprints", FINGERPRINTS),
        ("path list", newest_pathlist() or "(none)"),
    ]
    return "\n".join(
        "  %-12s %s%s" % (k, v, "" if os.path.exists(v) else "   (missing)")
        for k, v in rows
    )
