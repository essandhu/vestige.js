"""Drive ALL THREE references through one crowded sequence, to pin association semantics.

Cross-implementation oracle for
`packages/core/tests/validation/association-crowded-fixture.test.ts`.

The three existing sister fixtures (`sort-abewley/`, `ocsort-noahcao/`,
`bytetrack-foundationvision/`) all use well-separated boxes, so every cost matrix
they produce has an unambiguous optimum. That hides the fact that vestige's
association convention is not any of the three references' conventions -- the
conventions only coincide when the admissible cells already form an unambiguous
matching. This fixture is the one that puts them under competition.

The three conventions (see ADR-0005):

  sort.py        solve the FULL ungated IoU matrix maximizing IoU
                 (`linear_assignment(-iou_matrix)`), THEN drop matched pairs whose
                 IoU < iou_threshold.                        [solve-then-filter]

  OC_SORT        same, solving on -(iou + angle_diff_cost) but filtering on raw iou.
                 Secondary (BYTE/OCR) stages: same solve-then-filter, no shortcut.

  ByteTrack      `lap.lapjv(cost, extend_cost=True, cost_limit=match_thresh)`, which
                 minimizes  sum(matched cost) + match_thresh * (#unmatched ROWS),
                 with cells above match_thresh forbidden.    [rejection-cost]

vestige currently does neither: it gates cells beyond the cutoff to +Infinity and
runs a MAX-CARDINALITY min-cost assignment, which is forced to pair up tracks the
references would leave unmatched.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Setup (one-time): all three reference clones, plus requirements.txt.

    git clone https://github.com/abewley/sort.git ~/repos/sort
    git clone https://github.com/noahcao/OC_SORT.git ~/repos/OC_SORT
    git clone https://github.com/FoundationVision/ByteTrack.git ~/repos/ByteTrack
    cd packages/core/fixtures && pip install -r requirements.txt

Regenerate:

    python packages/core/fixtures/association-crowded/gen.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import scipy

SORT_PATH = Path(os.environ.get("SORT_PATH", str(Path.home() / "repos" / "sort")))
OC_SORT_PATH = Path(os.environ.get("OC_SORT_PATH", str(Path.home() / "repos" / "OC_SORT")))
BYTETRACK_PATH = Path(os.environ.get("BYTETRACK_PATH", str(Path.home() / "repos" / "ByteTrack")))

for _p, _name, _url in (
    (SORT_PATH, "sort", "https://github.com/abewley/sort.git"),
    (OC_SORT_PATH, "OC_SORT", "https://github.com/noahcao/OC_SORT.git"),
    (BYTETRACK_PATH, "ByteTrack", "https://github.com/FoundationVision/ByteTrack.git"),
):
    if not _p.exists():
        sys.exit(f"{_name} clone not found at {_p}.\n  git clone {_url} {_p}")

# --- Import rigging (see each sister fixture's gen.py for the per-reference story) ---
#
# numpy 2 removed the `np.float` alias; both sort.py and ByteTrack still use it.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

# sort.py imports matplotlib + skimage at module load for its CLI demo block.
_mpl = types.ModuleType("matplotlib")
_mpl.use = lambda *_a, **_kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("matplotlib", _mpl)
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("matplotlib.patches", types.ModuleType("matplotlib.patches"))
_skimage = types.ModuleType("skimage")
_skimage_io = types.ModuleType("skimage.io")
_skimage.io = _skimage_io  # type: ignore[attr-defined]
sys.modules.setdefault("skimage", _skimage)
sys.modules.setdefault("skimage.io", _skimage_io)

# ByteTrack: torch/cv2 are imported but unreachable on our path; yolox/__init__.py
# eagerly imports the whole training framework, so register the packages by hand.
for _n in ("torch", "torch.nn", "torch.nn.functional", "cv2"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["torch"].nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]  # type: ignore[attr-defined]
# scipy's array_api_compat sniffs `sys.modules['torch'].Tensor` when deciding whether
# an array is a torch tensor. A bare ModuleType has no `.Tensor`, and the AttributeError
# surfaces from deep inside scipy (which OC_SORT pulls in via filterpy). Give the stub
# a dummy class so the isinstance check simply returns False.
sys.modules["torch"].Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
for _pkg_name, _pkg_dir in (
    ("yolox", BYTETRACK_PATH / "yolox"),
    ("yolox.tracker", BYTETRACK_PATH / "yolox" / "tracker"),
):
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_pkg_dir)]  # type: ignore[attr-defined]
    sys.modules[_pkg_name] = _pkg

sys.path.insert(0, str(SORT_PATH))
sys.path.insert(0, str(OC_SORT_PATH))

from sort import KalmanBoxTracker, Sort  # noqa: E402
from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402


def _sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


SORT_CONFIG = {"max_age": 1, "min_hits": 3, "iou_threshold": 0.3}
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


class ByteArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False


BYTETRACK_CONFIG = {
    "track_thresh": ByteArgs.track_thresh,
    "track_buffer": ByteArgs.track_buffer,
    "match_thresh": ByteArgs.match_thresh,
    "mot20": ByteArgs.mot20,
    "frame_rate": 30,
}

IMG_INFO = (1080, 1920)
IMG_SIZE = (1080, 1920)

# --- Sequence design ------------------------------------------------------
#
# 60x60 boxes. NEIGHBOR_GAP is chosen so that two boxes one gap apart have
# IoU = (60-25)/(60+25) = 0.4118, and two boxes TWO gaps apart have
# IoU = (60-50)/(60+50) = 0.0909. Both sit far from every cutoff in play
# (SORT/OC-SORT iou_threshold 0.3; ByteTrack's effective gate of
# iou*score >= 0.2, i.e. iou >= 0.222 at score 0.9), so no assignment here is
# decided by a hair -- in particular, the cython_bbox "+1" convention that
# ByteTrack's IoU goes through (worth <= 0.006 on boxes this size) cannot flip
# any of them. The divergence this fixture pins is STRUCTURAL, not marginal.
BOX = 60.0
NEIGHBOR_GAP = 25.0  # IoU 0.4118 between neighbors
SCORE = 0.9

N_FRAMES = 60

# THE DISCRIMINATING STRUCTURE
# ----------------------------
# A pair of overlapping objects (LEAD and TRAIL, one gap apart). On the trigger
# frames, TRAIL is occluded AND a NEW object appears one gap on the far side of
# LEAD. The tracks are {LEAD, TRAIL}; the detections are {lead's box, new box}:
#
#                    det_lead   det_new
#     track LEAD       1.00      0.41
#     track TRAIL      0.41      0.09   <- below every threshold
#
# Every reference matches LEAD->det_lead, leaves TRAIL unmatched, and spawns a
# fresh track for det_new:
#   - sort/OC-SORT solve the FULL matrix (1.00 + 0.09 = 1.09 beats 0.41 + 0.41 =
#     0.82), then FILTER the 0.09 pair out.
#   - ByteTrack compares matching both (0.63 + 0.63 = 1.26) against matching
#     LEAD only and REJECTING TRAIL (0.10 + 0.80 = 0.90) and takes the latter.
#
# vestige's max-cardinality solver cannot leave TRAIL unmatched while an
# admissible cell remains, so it is forced into LEAD->det_new + TRAIL->det_lead:
# both tracks jump onto the wrong boxes and det_new never spawns. One frame of
# that permanently corrupts two ids.
#
# The sequence fires this twice, in two independent rows, so a single fluke
# cannot make the test pass.
#
# A NEW object must appear one gap from where LEAD *currently is*, not from LEAD's
# starting x -- `true_bbox` measures each track from its OWN first frame, so a late
# spawner starts at its literal x0 while LEAD has already drifted. Hence
# NEW.x0 = LEAD.x0 + LEAD.vx * NEW.first + NEIGHBOR_GAP.
LEAD_X0 = 300.0
LEAD_VX = 1.0
_new_x0 = lambda first: LEAD_X0 + LEAD_VX * first + NEIGHBOR_GAP  # noqa: E731

# The occlusion is exactly ONE frame. Two would exceed sort.py's max_age=1 and reap
# TRAIL entirely, which is legitimate reference behavior but would bury the
# association signal under a lifecycle event.
#
# name, x0, y0, vx, vy, first, last, occluded_frames
TRACKS_SPEC = [
    # Row 1 (y=100): LEAD_1 / TRAIL_1 pair; NEW_1 arrives at frame 20, the one frame
    # TRAIL_1 is occluded.
    ("LEAD_1", LEAD_X0, 100.0, LEAD_VX, 0.0, 0, 59, []),
    ("TRAIL_1", LEAD_X0 - NEIGHBOR_GAP, 100.0, LEAD_VX, 0.0, 0, 59, [20]),
    ("NEW_1", _new_x0(20), 100.0, LEAD_VX, 0.0, 20, 59, []),
    # Row 2 (y=400): same structure, later, so the two events cannot interfere.
    ("LEAD_2", LEAD_X0, 400.0, LEAD_VX, 0.0, 0, 59, []),
    ("TRAIL_2", LEAD_X0 - NEIGHBOR_GAP, 400.0, LEAD_VX, 0.0, 0, 59, [40]),
    ("NEW_2", _new_x0(40), 400.0, LEAD_VX, 0.0, 40, 59, []),
    # Control: isolated, always visible. Must keep one id in every implementation.
    ("SOLO", 900.0, 700.0, 1.0, 0.0, 0, 59, []),
]


def true_bbox(spec, frame: int) -> list[float]:
    _, x0, y0, vx, vy, first, _, _ = spec
    dt = frame - first
    x = x0 + vx * dt
    y = y0 + vy * dt
    return [x, y, x + BOX, y + BOX]


def build_detections(frame: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for spec in TRACKS_SPEC:
        _, _, _, _, _, first, last, occluded = spec
        if frame < first or frame > last or frame in occluded:
            continue
        rows.append(true_bbox(spec, frame) + [SCORE])
    return rows


def run_sort(frames_dets):
    KalmanBoxTracker.count = 0
    t = Sort(**SORT_CONFIG)
    out = []
    for dets_list in frames_dets:
        dets = (
            np.array(dets_list, dtype=np.float64)
            if dets_list
            else np.empty((0, 5), dtype=np.float64)
        )
        res = t.update(dets)
        out.append([[float(v) for v in row] for row in res])  # [x1,y1,x2,y2,id]
    return out


def run_ocsort(frames_dets):
    import trackers.ocsort_tracker.ocsort as _oc

    # OC_SORT's KalmanBoxTracker.count is class-level and survives across OCSort
    # instances; reset it so ids start at 1 deterministically.
    _oc.KalmanBoxTracker.count = 0
    t = OCSort(**OCSORT_CONFIG)
    out = []
    for dets_list in frames_dets:
        dets = (
            np.array(dets_list, dtype=np.float64)
            if dets_list
            else np.empty((0, 5), dtype=np.float64)
        )
        res = t.update(dets, img_info=IMG_INFO, img_size=IMG_SIZE)
        out.append([[float(v) for v in row] for row in res])  # [x1,y1,x2,y2,id]
    return out


def run_bytetrack(frames_dets):
    BaseTrack._count = 0
    t = BYTETracker(ByteArgs(), frame_rate=BYTETRACK_CONFIG["frame_rate"])
    out = []
    for dets_list in frames_dets:
        dets = (
            np.array(dets_list, dtype=np.float64)
            if dets_list
            else np.empty((0, 5), dtype=np.float64)
        )
        res = t.update(dets, IMG_INFO, IMG_SIZE)
        out.append(
            [
                [float(s.tlbr[0]), float(s.tlbr[1]), float(s.tlbr[2]), float(s.tlbr[3]), int(s.track_id)]
                for s in res
            ]
        )
    return out


def main() -> None:
    frames_dets = [build_detections(f) for f in range(N_FRAMES)]

    sort_out = run_sort(frames_dets)
    ocsort_out = run_ocsort(frames_dets)
    byte_out = run_bytetrack(frames_dets)

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/association-crowded/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "abewley_sha": _sha(SORT_PATH),
            "noahcao_sha": _sha(OC_SORT_PATH),
            "foundationvision_sha": _sha(BYTETRACK_PATH),
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "sort_config": SORT_CONFIG,
        "ocsort_config": OCSORT_CONFIG,
        "bytetrack_config": BYTETRACK_CONFIG,
        "sequence": {
            "n_frames": N_FRAMES,
            "box_size": BOX,
            "neighbor_gap": NEIGHBOR_GAP,
            "tracks_spec": [
                {
                    "name": n,
                    "x0": x0,
                    "y0": y0,
                    "vx": vx,
                    "vy": vy,
                    "first_frame_0indexed": fi,
                    "last_frame_0indexed": la,
                    "occluded_frames_0indexed": oc,
                }
                for (n, x0, y0, vx, vy, fi, la, oc) in TRACKS_SPEC
            ],
        },
        "frames": [
            {
                "frame_index": f + 1,
                "detections": frames_dets[f],
                "sort_out": sort_out[f],
                "ocsort_out": ocsort_out[f],
                "bytetrack_out": byte_out[f],
            }
            for f in range(N_FRAMES)
        ],
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {out_path} ({N_FRAMES} frames; "
        f"sort={sum(len(r) for r in sort_out)} ocsort={sum(len(r) for r in ocsort_out)} "
        f"bytetrack={sum(len(r) for r in byte_out)} track outputs)"
    )


if __name__ == "__main__":
    main()
