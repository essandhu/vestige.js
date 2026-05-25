"""Generate a per-frame trace of abewley/sort on a synthetic sequence.

Cross-implementation oracle for `packages/core/tests/validation/sort-abewley-fixture.test.ts`.
The TS test runs the same sequence through `SortTracker` and asserts that
the per-frame outputs match within tolerance.

Sister fixture to `ocsort-noahcao/`: same shape, same harness, different
reference. The point is to catch the same class of faithfulness drift in
SortTracker that the noahcao fixture caught in OcSortTracker.

This script imports the reference implementation from a user-provided clone
of `abewley/sort` (ARCHITECTURE.md §10.2 — reference implementations are
referenced, not vendored). Set `SORT_PATH` to override the default location.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output
JSON are committed together; never one without the other.

Setup (one-time):

    git clone https://github.com/abewley/sort.git ~/repos/sort
    git -C ~/repos/sort checkout 2236dff5019565958b84df7d871d41cc1db58ac7
    cd packages/core/fixtures
    pip install -r requirements.txt   # numpy + scipy + filterpy

Regenerate:

    python packages/core/fixtures/sort-abewley/gen.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

DEFAULT_SORT_PATH = Path.home() / "repos" / "sort"
SORT_PATH = Path(os.environ.get("SORT_PATH", str(DEFAULT_SORT_PATH)))

if not SORT_PATH.exists():
    sys.exit(
        f"sort clone not found at {SORT_PATH}.\n"
        f"Set SORT_PATH or clone: git clone https://github.com/abewley/sort.git {SORT_PATH}"
    )

# sort.py imports matplotlib + skimage at module load for its CLI demo block;
# the Sort class itself uses neither. Stub the modules out so this fixture's
# requirements.txt stays lean (no matplotlib, no scikit-image).
import types  # noqa: E402

_mpl = types.ModuleType("matplotlib")
_mpl.use = lambda *_a, **_kw: None
sys.modules.setdefault("matplotlib", _mpl)
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("matplotlib.patches", types.ModuleType("matplotlib.patches"))
_skimage = types.ModuleType("skimage")
_skimage_io = types.ModuleType("skimage.io")
_skimage.io = _skimage_io
sys.modules.setdefault("skimage", _skimage)
sys.modules.setdefault("skimage.io", _skimage_io)

sys.path.insert(0, str(SORT_PATH))

# Import only after sys.path is rigged. noinspection PyUnresolvedReferences below
# is for IDEs that don't understand the dynamic import path.
from sort import Sort, KalmanBoxTracker  # noqa: E402

# Lock the abewley commit hash into the fixture envelope. A mismatch with
# the README's pinned hash is what tells a future reviewer the reference
# version drifted.
ABEWLEY_SHA = subprocess.check_output(
    ["git", "-C", str(SORT_PATH), "rev-parse", "HEAD"], text=True
).strip()

# sort.py defaults (sort.py:Sort.__init__).
SORT_CONFIG = {
    "max_age": 1,
    "min_hits": 3,
    "iou_threshold": 0.3,
}

BOX_SIZE = 80.0
DET_SCORE = 0.9
N_FRAMES = 60

# Synthetic sequence: 6 tracks at well-separated start positions with distinct
# linear velocities, plus engineered single-frame occlusion windows that
# exercise the SortTracker `exportConfirmed` hit_streak gate — the same class
# of bug that PR #14 fixed in OcSortTracker.
#
# With sort.py defaults (max_age=1, min_hits=3) the only viable occlusion is
# single-frame: two consecutive misses → tsu=2 > max_age=1 → removed. The
# narrow window where the bug fires is the 3 frames immediately AFTER a
# single-frame miss on an already-confirmed track:
#
#   Frame N:    matched, hit_streak=k (large), tsu=0
#   Frame N+1:  miss → tsu=1, hit_streak still=k at export time (sort.py
#               resets hit_streak on the NEXT predict, after tsu>0)
#               → export gate: tsu<1 False → NOT output. State preserved.
#   Frame N+2:  match → tsu=0, hit_streak=1
#               → sort.py: 1 < min_hits(3) AND frame_count > 3 → NOT output
#               → buggy vestige: state==='confirmed' → output (BUG)
#   Frame N+3:  match → hit_streak=2 → still NOT output by sort.py
#   Frame N+4:  match → hit_streak=3 → both output
#
# So each single-frame occlusion buys 2 frames of bug-exposing output
# divergence. The sequence below has several of these windows spread across
# different tracks and frame ranges.
#
# Box size 80×80; positions chosen so no two tracks ever IoU-overlap (avoids
# Hungarian ties, keeps cross-implementation ID assignment deterministic).
TRACKS_SPEC = [
    # name, x0, y0, vx, vy, occlude_frames (0-indexed)
    ("A", 50.0, 50.0, 5.0, 0.0, []),                  # control: always visible
    ("B", 350.0, 50.0, 0.0, 5.0, [11]),               # one single-frame miss — bug window 12..13
    ("C", 50.0, 350.0, -3.0, 3.0, [20, 35]),          # two single-frame misses — bug windows 21..22 and 36..37
    ("D", 350.0, 350.0, 0.0, 0.0, []),                # control: stationary
    ("E", 600.0, 100.0, 2.0, 4.0, [25, 26]),          # two CONSECUTIVE misses → removed → reborn with new id
    ("F", 100.0, 550.0, 8.0, -2.0, [45]),             # late single-frame miss — bug window 46..47
]


def true_bbox(track_spec, frame: int) -> list[float]:
    _, x0, y0, vx, vy, _ = track_spec
    x = x0 + vx * frame
    y = y0 + vy * frame
    return [x, y, x + BOX_SIZE, y + BOX_SIZE]


def build_detections(frame: int) -> list[list[float]]:
    """Per-frame [[x1, y1, x2, y2, score], ...] honoring per-track occlusion windows."""
    rows: list[list[float]] = []
    for spec in TRACKS_SPEC:
        _, _, _, _, _, occlude = spec
        if frame in occlude:
            continue
        rows.append(true_bbox(spec, frame) + [DET_SCORE])
    return rows


def main() -> None:
    # KalmanBoxTracker.count is a class-level counter that survives across Sort
    # instances; reset it so the fixture's track ids start at 1 deterministically.
    KalmanBoxTracker.count = 0
    tracker = Sort(**SORT_CONFIG)
    frames: list[dict] = []
    for f in range(N_FRAMES):
        dets_list = build_detections(f)
        dets = (
            np.array(dets_list, dtype=np.float64)
            if dets_list
            else np.empty((0, 5), dtype=np.float64)
        )
        out = tracker.update(dets)
        # abewley returns np.ndarray of shape (M, 5): [x1, y1, x2, y2, id].
        # On empty frames returns np.empty((0, 5)).
        if isinstance(out, np.ndarray):
            out_rows = [[float(v) for v in row] for row in out]
        else:
            out_rows = []
        frames.append(
            {
                "frame_index": f + 1,
                "detections": dets_list,
                "tracks_out": out_rows,
            }
        )

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/sort-abewley/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "abewley_sha": ABEWLEY_SHA,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "sort_config": SORT_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX_SIZE,
            "tracks_spec": [
                {
                    "name": name,
                    "x0": x0,
                    "y0": y0,
                    "vx": vx,
                    "vy": vy,
                    "occlude_frames_0indexed": occlude,
                }
                for (name, x0, y0, vx, vy, occlude) in TRACKS_SPEC
            ],
        },
        "frames": frames,
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_path} ({N_FRAMES} frames, abewley={ABEWLEY_SHA[:8]})")


if __name__ == "__main__":
    main()
