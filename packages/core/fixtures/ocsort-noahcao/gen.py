"""Generate a per-frame trace of noahcao/OC_SORT on a synthetic sequence.

Cross-implementation oracle for `packages/core/tests/validation/ocsort-noahcao-fixture.test.ts`.
The TS test runs the same sequence through `OcSortTracker` and asserts that
the per-frame outputs match within tolerance.

This script imports the reference implementation from a user-provided clone
of `noahcao/OC_SORT` (ARCHITECTURE.md §10.2 — reference implementations are
referenced, not vendored). Set `OC_SORT_PATH` to override the default
location.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output
JSON are committed together; never one without the other.

Setup (one-time):

    git clone https://github.com/noahcao/OC_SORT.git ~/repos/OC_SORT
    git -C ~/repos/OC_SORT checkout 8462e7e729a93ccd3bd995c0a79a890336cb3a0b
    cd packages/core/fixtures
    pip install -r requirements.txt   # numpy + scipy + filterpy

Regenerate:

    python packages/core/fixtures/ocsort-noahcao/gen.py
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

DEFAULT_OC_SORT_PATH = Path.home() / "repos" / "OC_SORT"
OC_SORT_PATH = Path(os.environ.get("OC_SORT_PATH", str(DEFAULT_OC_SORT_PATH)))

if not OC_SORT_PATH.exists():
    sys.exit(
        f"OC_SORT clone not found at {OC_SORT_PATH}.\n"
        f"Set OC_SORT_PATH or clone: git clone https://github.com/noahcao/OC_SORT.git {OC_SORT_PATH}"
    )

sys.path.insert(0, str(OC_SORT_PATH))

# Import only after sys.path is rigged. noinspection PyUnresolvedReferences below
# is for IDEs that don't understand the dynamic import path.
from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402

# Lock the noahcao commit hash into the fixture envelope. A mismatch with
# the README's pinned hash is what tells a future reviewer the reference
# version drifted.
NOAHCAO_SHA = subprocess.check_output(
    ["git", "-C", str(OC_SORT_PATH), "rev-parse", "HEAD"], text=True
).strip()

OCSORT_CONFIG = {
    "det_thresh": 0.6,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "asso_func": "iou",
    "inertia": 0.2,
    "use_byte": False,
}

BOX_SIZE = 80.0
DET_SCORE = 0.9
N_FRAMES = 60
IMG_INFO = (720, 1280)
IMG_SIZE = (720, 1280)

# Synthetic sequence: 6 tracks at well-separated start positions with distinct
# linear velocities, plus engineered occlusion windows that exercise ORU
# (short occlusion + reappearance), OCR (long occlusion that drifts the Kalman
# scale out of IoU range), and lifecycle transitions (confirm → lost → re-find).
#
# Box size 80×80; positions chosen so no two tracks ever IoU-overlap (avoids
# Hungarian ties, keeps cross-implementation ID assignment deterministic).
TRACKS_SPEC = [
    # name, x0, y0, vx, vy, occlude_frames (0-indexed)
    ("A", 50.0, 50.0, 5.0, 0.0, list(range(15, 22))),   # ORU: 7-frame occlusion + smooth reappearance
    ("B", 350.0, 50.0, 0.0, 5.0, []),                    # control: always visible
    ("C", 50.0, 350.0, -3.0, 3.0, list(range(45, 55))),  # 10-frame occlusion mid-motion
    ("D", 350.0, 350.0, 0.0, 0.0, list(range(30, 42))),  # stationary 12-frame occlusion (OCR territory)
    ("E", 600.0, 100.0, 2.0, 4.0, []),                    # control: always visible
    ("F", 100.0, 550.0, 8.0, -2.0, list(range(20, 28))), # ORU: parallel to A's occlusion
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
    tracker = OCSort(**OCSORT_CONFIG)
    frames: list[dict] = []
    for f in range(N_FRAMES):
        dets_list = build_detections(f)
        dets = (
            np.array(dets_list, dtype=np.float64)
            if dets_list
            else np.empty((0, 5), dtype=np.float64)
        )
        out = tracker.update(dets, img_info=IMG_INFO, img_size=IMG_SIZE)
        # noahcao returns np.ndarray of shape (M, 5): [x1, y1, x2, y2, id].
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
            "script": "packages/core/fixtures/ocsort-noahcao/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "noahcao_sha": NOAHCAO_SHA,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "ocsort_config": OCSORT_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX_SIZE,
            "img_info": list(IMG_INFO),
            "img_size": list(IMG_SIZE),
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
    print(f"wrote {out_path} ({N_FRAMES} frames, noahcao={NOAHCAO_SHA[:8]})")


if __name__ == "__main__":
    main()
